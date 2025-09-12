import copy

import torch.nn.functional as F
import math
from logging import DEBUG

import ray
import torch
from flwr.common import logger
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor

from clients.DataPoisoning_client import DataPoisoningClient
from utils import get_device

"""
Chameleon Attack, based on:

Y. Dai and S. Li, “Chameleon: Adapting to Peer Images for Planting Durable Backdoors in Federated Learning,” in Proceedings of the 40th International Conference on Machine Learning, PMLR, Jul. 2023, pp. 6712–6725. Accessed: Mar. 08, 2024. [Online]. Available: https://proceedings.mlr.press/v202/dai23a.html

Original Code: https://github.com/ybdai7/Chameleon-durable-backdoor
"""


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: """

    def __init__(self):
        super(SupConLoss, self).__init__()
        self.temperature = 0.07
        self.base_temperature = 0.07

    def forward(self, features, labels, fac_label: int, inter_labels: int, poison_per_batch=None, scale_weight=1):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            fac_label: Label of the facilitator, i.e., the backdoor target class
            inter_labels: Labels of the interferers, i.e., the source classes
            poison_per_batch: Portion of the batch that is poisoned
            scale_weight: Parameter beta in the paper
        Returns:
            A loss scalar.
        """
        device = get_device()

        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)

        mask = torch.eq(labels, labels.T).float().to(device)
        mask_scale = mask.clone().detach()
        mask_cross_feature = torch.ones_like(mask_scale).to(device)

        label_flatten = labels.view(-1)
        for ind, (label, inter_label) in enumerate(zip(labels.view(-1), inter_labels)):
            # Scale by beta, if the image is a facilitator
            if label == fac_label:
                mask_scale[ind, :] = mask[ind, :] * scale_weight
            #
            if ind < int(poison_per_batch * batch_size):
                label_inter_row_eq = torch.eq(label_flatten, inter_label).float().to(device)
                label_inter_row_nq = torch.ne(label_flatten, inter_label).float().to(device)
                label_inter_row = label_inter_row_eq + label_inter_row_nq
                mask_cross_feature[ind, :] = mask_cross_feature[ind, :] * label_inter_row
            if label == inter_label:
                mask_cross_feature[ind, 0:int(poison_per_batch * batch_size)] = mask_cross_feature[ind, 0:int(
                    poison_per_batch * batch_size)]

        contrast_feature = features
        anchor_feature = features

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature) * mask_cross_feature
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask
        mask_scale = mask_scale * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos_mask = (mask_scale * log_prob).sum(1)
        mask_check = mask.sum(1)
        for ind, mask_item in enumerate(mask_check):
            if mask_item == 0:
                continue
            else:
                mask_check[ind] = 1 / mask_item
        mask_apply = mask_check
        mean_log_prob_pos = mean_log_prob_pos_mask * mask_apply
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(batch_size).mean()

        return loss


SupConLoss = SupConLoss().cuda()


class ChameleonClient(DataPoisoningClient):
    def train(self, trainloader, current_round, configuration: dict = None, malicious: bool = None):
        if malicious:
            target_model = copy.deepcopy(self.net)
            target_params_variables = {name: param.clone().detach().requires_grad_(False) for name, param in
                                       target_model.named_parameters()}

            current_number_of_adversaries = ray.get(self.client_communicator.get_malicious.remote())

            self.net.train()

            contrastive_model = self.contrastive_learning(configuration, target_model, target_params_variables,
                                                          trainloader)

            # Copy the parameters of the first layers into our main model
            encoder_state_dict = contrastive_model.state_dict()
            dst_state_dict = self.net.state_dict()

            for name, param in encoder_state_dict.items():
                if name in dst_state_dict:
                    dst_state_dict[name].copy_(param)

            # Freezing the first layers that were trained using contrastive learning
            for params in self.net.named_parameters():
                if params[0] in dict(contrastive_model.named_parameters()):
                    params[1].requires_grad = False

            self.poisoning_training(configuration, target_model, target_params_variables, trainloader)

            logger.log(DEBUG, f'Global model norm: {model_global_norm(target_model)}.')
            logger.log(DEBUG, f'Norm before scaling: {model_global_norm(self.net)}. '
                             f'Distance: {model_dist_norm(self.net, target_params_variables)}')

            # We scale data according to formula: L = 100*X-99*G = G + (100*X- 100*G).
            clip_rate = (configuration['attacker']['gamma'] / current_number_of_adversaries)
            logger.log(DEBUG, f"Scaling by  {clip_rate}")
            self.net = self.scale_model_update(self.net, target_model, clip_rate)
            distance = model_dist_norm(self.net, target_params_variables)
            logger.log(DEBUG, f'Scaled Norm after poisoning: {model_global_norm(self.net)}, distance: {distance}')

            # Perform PGD after scaling again
            self.net = project_onto_ball(self.net, configuration['attacker']['norm_bound'], target_model, target_params_variables)
            distance = model_dist_norm(self.net, target_params_variables)
            logger.log(DEBUG,
                       f'Scaled Norm after poisoning and clipping: '
                       f'{model_global_norm(self.net)}, distance: {distance}')


            distance = model_dist_norm(self.net, target_params_variables)
            logger.log(DEBUG, f"Total norm for {current_number_of_adversaries} "
                             f"adversaries is: {model_global_norm(self.net)}. distance: {distance}")
            logger.log(DEBUG, f"Loss: {self.metric.clean_loss}, Accuracy: {self.metric.clean_accuracy}")
        else:
            super().train(trainloader, current_round, configuration, malicious)

    def poisoning_training(self, configuration, target_model, target_params_variables, trainloader):
        poison_optimizer, lr_scheduler = self.create_optimizer(configuration, True)
        epochs = configuration['local_epochs_malicious_clients']
        self.metric.reset()
        self.test(self.trainloader, configuration)
        acc_initial, acc_p = self.metric.clean_accuracy, self.metric.backdoor_accuracy

        logger.log(DEBUG, f"Starting poisoning training for {epochs} epochs...")
        for internal_epoch in range(1, epochs + 1):
            for batch in trainloader:
                batch, _ = self.backdoor.get_poisoned_batch(batch, self.task.backdoor_train_dataset,
                                                            poison_ratio=configuration['backdoor']['poison_ratio'])

                poison_optimizer.zero_grad()
                data, labels = batch["img"].to(self.device), batch["label"].to(self.device)
                output = self.net(data)
                loss = nn.functional.cross_entropy(output, labels)  # We removed the distance loss, as it was not used in the Chameleon paper
                loss.backward()

                poison_optimizer.step()
                # Perform PGD
                self.net = project_onto_ball(self.net, configuration['attacker']['norm_bound'], target_model,
                                             target_params_variables)

            if lr_scheduler is not None:
                lr_scheduler.step()

            self.metric.reset()
            self.test(self.trainloader, configuration)

            if self.metric.backdoor_loss <= 0.0001:
                if self.metric.clean_accuracy < acc_initial:
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    continue
                logger.log(DEBUG, 'Converged earlier')
                break
            logger.log(DEBUG, f'Distance: {model_dist_norm(self.net, target_params_variables)}')

    def contrastive_learning(self, configuration, target_model, target_params_variables, trainloader):
        epochs = configuration['attacker']['contrastive_local_epochs']
        contrastive_model = create_feature_extractor(self.net, {configuration['attacker']['split_layer']: 'embedding'})

        if 'local_lr' in configuration['attacker']['contrastive_optimizer']:
            contrastive_lr = configuration['attacker']['contrastive_optimizer']['local_lr']
        elif 'lr_factor' in configuration['attacker']['contrastive_optimizer'] and configuration['attacker']['contrastive_optimizer']['lr_factor'] is not None:
            contrastive_lr = configuration['optimizer']['local_lr'] * configuration['attacker']['contrastive_optimizer']['lr_factor']
        poison_optimizer_contrastive = torch.optim.SGD(
            filter(lambda p: p.requires_grad, contrastive_model.parameters()),
            lr=contrastive_lr,
            momentum=configuration['attacker']['contrastive_optimizer']['momentum'],
            weight_decay=configuration['attacker']['contrastive_optimizer']['weight_decay'])
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            poison_optimizer_contrastive,
            milestones=configuration['attacker']['contrastive_lr_scheduler']['milestones'],
            gamma=configuration['attacker']['contrastive_lr_scheduler']['gamma'])

        logger.log(DEBUG, f"Starting contrastive training for {epochs} epochs...")
        for _ in range(1, epochs + 1):
            for batch in trainloader:
                batch, _ = self.backdoor.get_poisoned_batch(batch, self.task.backdoor_train_dataset,
                                                            poison_ratio=configuration['attacker'][
                                                                'contrastive_poison_ratio'])

                data, labels = batch["img"].to(self.device), batch["label"].to(self.device)

                # Extract the embedding representation of the input data
                output = contrastive_model(data)['embedding']
                output = F.adaptive_avg_pool2d(output, (1, 1))
                output = torch.flatten(output, 1)
                output = F.normalize(output, dim=1)

                contrastive_loss = SupConLoss(output, labels, self.backdoor.target_class,
                                              [self.backdoor.source_class] * labels.shape[0] if self.backdoor.source_class else labels,
                                              poison_per_batch=configuration['attacker']['contrastive_poison_ratio'],
                                              scale_weight=configuration['attacker']['beta'])

                contrastive_loss.backward()

                poison_optimizer_contrastive.step()

                # Perform PGD (this method unfortunately destroys the feature_extractor mapping, thus, we have to call create_feature_extractor again
                contrastive_model = create_feature_extractor(project_onto_ball(self.net, configuration['attacker']['norm_bound'], target_model, target_params_variables), {configuration['attacker']['split_layer']: 'embedding'})

            if lr_scheduler is not None:
                lr_scheduler.step()

        return contrastive_model

    def scale_model_update(self, model, target_model, scaling_factor):
        for key, value in model.state_dict().items():
            if key in self.get_variable_params():
                target_value = target_model.state_dict()[key]
                new_value = target_value + (value - target_value) * scaling_factor
                model.state_dict()[key].copy_(new_value)
        return model


def model_dist_norm(model, target_params):
    squared_sum = 0
    for name, layer in model.named_parameters():
        squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
    return math.sqrt(squared_sum)


def model_global_norm(model):
    squared_sum = 0
    for name, layer in model.named_parameters():
        squared_sum += torch.sum(torch.pow(layer.data, 2))
    return math.sqrt(squared_sum)


def project_onto_ball(model, norm_bound, target_model, target_params_variables):
    model_norm = model_dist_norm(model, target_params_variables)
    if model_norm > norm_bound:
        logger.log(DEBUG, f'The limit reached for distance: '
                         f'{model_dist_norm(model, target_params_variables)}')
        norm_scale = norm_bound / model_norm
        for name, layer in model.named_parameters():
            if '__' in name:
                continue
            clipped_difference = norm_scale * (layer.data - target_model.state_dict()[name])
            layer.data.copy_(target_model.state_dict()[name] + clipped_difference)
    return model
