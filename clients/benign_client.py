import json
from collections import OrderedDict
from typing import List
from logging import DEBUG

import flwr as fl
import numpy as np
import ray
import torch
import torch.nn.functional as F
from flwr.common import logger

from backdoors import Backdoor
from tasks.Metrics import ImageClassificationMetric
from tasks.Task import Task
from utils import get_device, fix_random, is_method_implemented


class BenignClient(fl.client.NumPyClient):
    def __init__(self, cid, task: Task, backdoor: Backdoor, experiment_descriptor: str, reproducible: bool=False, seed: int=42):
        self.cid = cid
        self.task = task
        self.net = task.get_model()
        self.trainloader = task.get_dataloader(self.cid, "train")
        self.client_communicator = ray.get_actor("client_communicator", namespace="clients")
        self.backdoor = backdoor
        self.device = get_device()
        self.metric = ImageClassificationMetric()

        if reproducible:
            global_round = ray.get(self.client_communicator.get_something.remote('global_round'))
            fix_random(seed + (global_round if global_round else 0))

        # Setup logger
        logger.configure(f'Client {self.cid}', f"./logs/{experiment_descriptor}/log.log")
        # Store the logging handler that this call to `configure` created, to be able to remove it when the client is
        # destroyed
        self.logging_handler = logger.logger.handlers[-1]

    def __del__(self):
        # As clients are dynamically spawned when required, we have to remove the logging handlers when a client is
        # destroyed. Otherwise, there would be one logging handler more per round per client.
        logger.logger.removeHandler(self.logging_handler)

        # Explicitly remove dataloader
        del self.trainloader
        import gc
        gc.collect()

    def get_variable_params(self) -> List[str]:
        return list(filter(lambda name: 'num_batches_tracked' not in name, self.net.state_dict().keys()))

    def get_parameters(self, config) -> List[np.ndarray]:
        with torch.no_grad():
            params = self.get_variable_params()
            state_dict = self.net.state_dict()
            return [state_dict[name].cpu().numpy() for name in params]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Load the dictionary of trainable parameters
        params_dict = zip(self.get_variable_params(), parameters)
        state_dict = OrderedDict({k: torch.as_tensor(v, device=self.device) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train(self.trainloader, config['current_round'], configuration=json.loads(config['parameters']))
        return self.get_parameters(config), len(self.trainloader), self.metric.to_dict()

    def evaluate(self, parameters, config):
        configuration = json.loads(config['parameters'])
        self.set_parameters(parameters)
        self.test(self.trainloader, configuration)
        return self.metric.clean_loss, len(self.trainloader), self.metric.to_dict()

    def process_batch(self, batch, train, optimizer=None, alternative_labels=None):
        images, labels = batch["img"].to(self.device), batch["label"].to(self.device)
        outputs = self.net(images)
        loss = self.task.criterion(outputs, labels)
        with torch.no_grad(): # Prevent gradient tracking for accuracy computation
            preds = outputs.argmax(dim=1)
            correct = (preds == labels).sum().item()
            confidence = F.softmax(outputs, dim=1)[torch.arange(labels.shape[0]), labels].sum().item()

            alternative_loss, alternative_correct, alternative_confidence = 0.0, 0, 0.0
            if alternative_labels is not None:
                alternative_labels = alternative_labels.to(self.device)
                alternative_loss = self.task.criterion(outputs, alternative_labels).item()
                alternative_correct = (preds == alternative_labels).sum().item()
                alternative_confidence = F.softmax(outputs, dim=1)[torch.arange(alternative_labels.shape[0]), alternative_labels].sum().item()

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss.item(), correct, confidence, alternative_loss, alternative_correct, alternative_confidence

    def train(self, trainloader, current_round, configuration: dict = None):
        """Train the network on the training set."""
        optimizer, lr_scheduler = self.create_optimizer(configuration)
        epochs = configuration['local_epochs']
        self.net.train()

        logger.log(DEBUG, f"Starting benign training for {epochs} epochs...")

        for epoch in range(epochs):
            self.metric.reset()
            for batch_idx, batch in enumerate(trainloader):
                batch_loss, batch_correct, batch_confidence, _, _, _ = self.process_batch(batch, train=True, optimizer=optimizer)
                self.metric.update_from_batch(batch_loss, batch_correct, len(batch['label']), batch_confidence, 'clean')

            if lr_scheduler is not None:
                lr_scheduler.step()

            if epoch == epochs - 1:  # Return accuracy and loss of the last epoch
                logger.log(DEBUG, f"Loss: {self.metric.clean_loss}, Accuracy: {self.metric.clean_accuracy}, Confidence: {self.metric.clean_confidence}")

    def create_optimizer(self, configuration: dict = None, malicious: bool = False):
        key = 'adversarial_optimizer' if malicious else 'optimizer'

        # If adversary does not specify learning rate, take the benign lr and multiply it by lr_factor.
        # If that's not given, simply copy the benign lr
        local_lr = configuration['optimizer']['local_lr']
        if malicious:
            if 'local_lr' in configuration[key] and configuration[key]['local_lr'] is not None:
                local_lr = configuration[key]['local_lr']
            elif 'lr_factor' in configuration['adversarial_optimizer'] and configuration['adversarial_optimizer']['lr_factor'] is not None:
                local_lr *= configuration['adversarial_optimizer']['lr_factor']

        if configuration[key]['name'].lower() == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=local_lr, momentum=configuration[key]['momentum'], weight_decay=configuration[key]['weight_decay'])
        elif configuration[key]['name'].lower() == "adam":
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=local_lr, weight_decay=configuration[key]['weight_decay'])
        else:
            raise ValueError(f"Unrecognized optimizer: {configuration[key]['name']}")

        key = 'adversarial_lr_scheduler' if malicious else 'lr_scheduler'

        epochs = configuration['local_epochs_malicious_clients'] if malicious else configuration['local_epochs']
        if not configuration[key]['name']:
            lr_scheduler = None
        elif configuration[key]['name'].lower() == "multi_step":
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(a * epochs) for a in configuration[key]['milestones']], gamma=configuration[key]['gamma'])
        elif configuration[key]['name'].lower() == "step":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration[key]['step_size'], gamma=configuration[key]['gamma'])
        elif configuration[key]['name'].lower() == "exponential":
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=configuration[key]['gamma'])
        elif configuration[key]['name'].lower() == "cosine_annealing":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(configuration[key]['T_max']), eta_min=float(configuration[key]['eta_min']))
        else:
            raise ValueError(f"Unrecognized scheduler: {configuration[key]['name']}")
        return optimizer, lr_scheduler

    def test(self, testloader, configuration):
        """Evaluate the network on the entire test set."""
        self.net.eval()

        is_semantic_backdoor = len(self.task.backdoor_test_indices.keys()) > 0

        with torch.no_grad():
            self.metric.reset()
            for batch in testloader:
                benign_batch_loss, benign_batch_correct, benign_batch_confidence, _, _, _ = self.process_batch(batch, train=False)
                self.metric.update_from_batch(benign_batch_loss, benign_batch_correct, len(batch['label']), benign_batch_confidence, 'clean')

                if is_semantic_backdoor:
                    backdoored_batch, backdoored_batch_with_original_labels = self.backdoor.get_poisoned_batch(batch, self.task.backdoor_test_dataset, poison_ratio=1)
                else:
                    backdoored_batch, backdoored_batch_with_original_labels = self.backdoor.get_poisoned_batch(batch, None, poison_ratio=1)

                backdoor_batch_loss, backdoor_batch_correct, backdoor_batch_confidence, alternative_batch_loss, alternative_batch_correct, alternative_batch_confidence = self.process_batch(backdoored_batch, train=False, alternative_labels=backdoored_batch_with_original_labels['label'])

                self.metric.update_from_batch(backdoor_batch_loss, backdoor_batch_correct, len(batch['label']), backdoor_batch_confidence, 'backdoor')
                self.metric.update_from_batch(alternative_batch_loss, alternative_batch_correct, len(batch['label']), alternative_batch_confidence, 'backdoored')
