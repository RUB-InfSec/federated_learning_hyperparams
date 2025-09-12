import copy
from logging import DEBUG
from time import sleep
from flwr.common import logger

import ray
import torch

from PoisoningScheduler import PoisoningScheduler
from backdoors.Backdoor import Backdoor
from clients.DataPoisoning_client import DataPoisoningClient
from tasks.Task import Task

"""
A3FL attack, based on:

H. Zhang, J. Jia, J. Chen, L. Lin, and D. Wu, “A3FL: Adversarially Adaptive Backdoor Attacks to Federated Learning,” presented at the Neural Information Processing Systems, 2023. Accessed: Mar. 14, 2024. [Online]. Available: https://www.semanticscholar.org/paper/A3FL%3A-Adversarially-Adaptive-Backdoor-Attacks-to-Zhang-Jia/a5ee25f920ba92d7d89ec7789a820d6da31a4f3a

Original Code: https://github.com/hfzhang31/a3fl
"""


class A3FLClient(DataPoisoningClient):
    def __init__(self, cid, task: Task, backdoor: Backdoor, experiment_descriptor,
                 poisoning_scheduler: PoisoningScheduler, reproducible=False, seed=42):
        super().__init__(cid, task, backdoor, experiment_descriptor, poisoning_scheduler, reproducible, seed)
        self.latest_grads = {
            name: copy.deepcopy(dict(self.net.named_parameters())[name].grad) for name in
            dict(self.net.named_parameters())
        }

    def train(self, trainloader, current_round, configuration: dict = None, malicious: bool = None):
        if malicious and ray.get(self.client_communicator.get_active_id.remote()) == self.cid:
            global_model_copy = copy.deepcopy(self.net.state_dict())
            # This is just an ugly hack to instantiate gradients in self.net as these are required in the
            # `search_trigger` method (or more precisely: in the get_adv_model method)
            super().train(self.trainloader, current_round, configuration, malicious)

            logger.log(DEBUG, "Optimizing trigger...")
            self.search_trigger(self.net, self.trainloader, configuration)
            self.client_communicator.store_something.remote(f'done_trigger_search_{current_round}', True)

            # Reset self.net for training with optimized trigger afterward
            self.net.load_state_dict(global_model_copy)
        elif malicious:
            logger.log(DEBUG, "Waiting for trigger optimization...")
            while ray.get(self.client_communicator.get_something.remote(f'done_trigger_search_{current_round}')) is False:
                sleep(1)

        super().train(self.trainloader, current_round, configuration, malicious)

    def get_adv_model(self, model, dl, trigger, mask, configuration):
        adv_model = copy.deepcopy(model)
        adv_model.train()
        ce_loss = torch.nn.CrossEntropyLoss()

        # ~ Algorithm 1, line 10
        adv_opt = torch.optim.SGD(adv_model.parameters(), lr=configuration['attacker']['alpha_2'], momentum=0.9, weight_decay=5e-4)
        for _ in range(configuration['attacker']['K']):  # = config.dm_adv_epochs in original project
            for batch in dl:
                inputs, labels = batch['img'], batch['label']
                inputs = trigger * mask + (1 - mask) * inputs
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = adv_model(inputs)
                loss = ce_loss(outputs, labels)
                adv_opt.zero_grad()
                loss.backward()
                adv_opt.step()

        model_params = dict(model.named_parameters())  # Cache parameters
        adv_params = dict(adv_model.named_parameters())
        cos_loss = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
        # Vectorized similarity computation
        grads = [
            cos_loss(adv_params[name].grad.flatten(), model_params[name].grad.flatten())
            for name in adv_params if 'conv' in name
        ]
        return adv_model, torch.stack(grads).mean()

    def search_trigger(self, model, dataloader, configuration: dict):
        model.eval()

        ce_loss = torch.nn.CrossEntropyLoss()
        alpha_1 = configuration['attacker']['alpha_1']

        trigger = ray.get(self.client_communicator.get_something.remote('trigger')).clone()
        normal_grad = 0.

        adv_model = None
        adv_w = 0.0
        for i in range(configuration['attacker']['K_trigger']):
            if i != 0:
                adv_model, adv_w = self.get_adv_model(model, dataloader, trigger, self.backdoor.mask, configuration)

            for batch in dataloader:
                trigger.requires_grad_()

                inputs = (trigger * self.backdoor.mask + batch['img'] * (1 - self.backdoor.mask)).to(self.device)
                labels = torch.full_like(batch['label'], self.backdoor.target_class, device=self.device)

                # Algorithm 1, line 6
                loss = ce_loss(model(inputs), labels)

                if adv_model is not None:
                    nm_loss = ce_loss(adv_model(inputs), labels)

                    if loss is None:
                        loss = configuration['attacker']['lambda_0'] * adv_w * nm_loss
                    else:
                        loss += configuration['attacker']['lambda_0'] * adv_w * nm_loss

                # Adapt trigger (Algorithm 1, line 7)
                if loss is not None:
                    loss.backward()
                    normal_grad += trigger.grad.sum()
                    trigger = (trigger - alpha_1 * trigger.grad.sign()).detach().clamp(-2, 2)
                    trigger.requires_grad_()

        trigger = trigger.detach()

        self.client_communicator.store_something.remote('trigger', trigger)
