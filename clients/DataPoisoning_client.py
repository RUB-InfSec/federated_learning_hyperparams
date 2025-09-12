import json
from logging import DEBUG

from flwr.common import logger

from PoisoningScheduler import PoisoningScheduler
from backdoors import Backdoor
from clients.benign_client import BenignClient
from tasks.Task import Task


class DataPoisoningClient(BenignClient):
    def __init__(self, cid, task: Task, backdoor: Backdoor, experiment_descriptor,
                 poisoning_scheduler: PoisoningScheduler, reproducible=False, seed=42):
        super().__init__(cid, task, backdoor, experiment_descriptor, reproducible, seed)
        self.poisoning_scheduler = poisoning_scheduler

    def fit(self, parameters, config):
        configuration = json.loads(config['parameters'])

        malicious = self.poisoning_scheduler.poison(self.cid)
        if malicious:
            if 'batch_size' in configuration['attacker'] and configuration['batch_size'] != configuration['attacker']['batch_size']:
                self.trainloader = self.task.get_dataloader(self.cid, 'train', configuration['attacker']['batch_size'])

            self.set_parameters(parameters)
            self.train(self.trainloader, config['current_round'], configuration=configuration, malicious=malicious)
            return self.get_parameters(self.net), len(self.trainloader), self.metric.to_dict()
        else:
            return super().fit(parameters, config)


    def train(self, trainloader, current_round, configuration: dict = None, malicious: bool = None):
        if malicious:
            optimizer, lr_scheduler = self.create_optimizer(configuration, malicious)
            epochs = configuration['local_epochs_malicious_clients']
            self.net.train()

            logger.log(DEBUG, f"Starting malicious training for {epochs} epochs...")
            for epoch in range(epochs):
                self.metric.reset()
                for batch in trainloader:
                    batch, _ = self.backdoor.get_poisoned_batch(batch, self.task.backdoor_train_dataset)
                    batch_loss, batch_correct, batch_confidence, _, _, _ = self.process_batch(batch, train=True, optimizer=optimizer)
                    self.metric.update_from_batch(batch_loss, batch_correct, len(batch['label']), batch_confidence, 'backdoor')

                if lr_scheduler is not None:
                    lr_scheduler.step()

                if epoch == epochs - 1:  # Return accuracy and loss of the last epoch
                    logger.log(DEBUG, f"Loss: {self.metric.backdoor_loss}, Accuracy: {self.metric.backdoor_accuracy}, Confidence: {self.metric.backdoor_confidence}")
        else:
            super().train(trainloader, current_round, configuration)
