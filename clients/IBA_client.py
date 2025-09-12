import copy
import time
from logging import INFO

import numpy as np
import ray
from flwr.common import logger

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn.functional as F

from PoisoningScheduler import PoisoningScheduler
from backdoors.Backdoor import Backdoor
from clients.DataPoisoning_client import DataPoisoningClient
from tasks.Task import Task
from utils import get_device

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_MIN = ((np.array([0, 0, 0]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).min()
IMAGENET_MAX = ((np.array([1, 1, 1]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).max()


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self, out_channel):
        super().__init__()

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Sequential(
            nn.Conv2d(64, out_channel, 1),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        out = torch.tanh(out)

        return out


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def create_trigger_model(dataset, attack_model=None):
    """ Create trigger model """
    if dataset == 'cifar10':
        atkmodel = UNet(3).to(get_device())
    elif dataset == 'mnist':
        atkmodel = Autoencoder().to(get_device())
    elif dataset == 'tiny-imagenet' or dataset == 'tiny-imagenet32' or dataset == 'gtsrb':
        if attack_model is None:
            atkmodel = Autoencoder().to(get_device())
        elif attack_model == 'unet':
            atkmodel = UNet(3).to(get_device())
    else:
        raise Exception(f'Invalid atk model {dataset}')

    return atkmodel


def all2one_target_transform(x, attack_target=1):
    return torch.ones_like(x) * attack_target


def exponential_decay(init_val, decay_rate, t):
    return init_val * (1.0 - decay_rate) ** t


def proj_l_inf(model, model_original, pgd_eps):
    w = list(model.parameters())
    # adversarial learning rate
    eta = 0.001
    for i in range(len(w)):
        # uncomment below line to restrict proj to some layers
        if True:  # i == 6 or i == 8 or i == 10 or i == 0 or i == 18:
            w[i].data = w[i].data - eta * w[i].grad.data
            # projection step
            m1 = torch.lt(torch.sub(w[i], model_original[i]), -pgd_eps)
            m2 = torch.gt(torch.sub(w[i], model_original[i]), pgd_eps)
            w1 = (model_original[i] - pgd_eps) * m1
            w2 = (model_original[i] + pgd_eps) * m2
            w3 = (w[i]) * (~(m1 + m2))
            wf = w1 + w2 + w3
            w[i].data = wf.data


def get_clip_image(dataset="cifar10"):
    if dataset in ['tiny-imagenet', 'tiny-imagenet32']:
        def clip_image(x):
            return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
    elif dataset == 'cifar10':
        def clip_image(x):
            return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
    elif dataset == 'mnist':
        def clip_image(x):
            return torch.clamp(x, -1.0, 1.0)
    else:
        raise Exception(f'Invalid dataset: {dataset}')
    return clip_image


class IBAClient(DataPoisoningClient):
    def __init__(self, cid, task: Task, backdoor: Backdoor, experiment_descriptor,
                 poisoning_scheduler: PoisoningScheduler, reproducible=False, seed=42):
        super().__init__(cid, task, backdoor, experiment_descriptor, poisoning_scheduler, reproducible, seed)
        self.retrain = ray.get(self.client_communicator.get_something.remote('retrain'))
        self.alternative_training = ray.get(self.client_communicator.get_something.remote('alternative_training'))
        self.local_acc_poison = ray.get(self.client_communicator.get_something.remote('local_acc_poison'))
        self.cur_training_eps = ray.get(self.client_communicator.get_something.remote('cur_training_eps'))
        self.start_decay_r = ray.get(self.client_communicator.get_something.remote('start_decay_r'))
        self.atkmodel_optimizer = None

    def train(self, trainloader, current_round, configuration: dict = None, malicious: bool = None):
        if malicious:
            # Initialize default values
            self.atkmodel = create_trigger_model(self.task.dataset_name, attack_model=configuration['attacker']['attack_model'])

            if atkmodel_state_dict := ray.get(self.client_communicator.get_something.remote('atkmodel')):
                self.atkmodel.load_state_dict(atkmodel_state_dict)

            self.atkmodel.to(self.device)

            self.atkmodel_optimizer = torch.optim.Adam(self.atkmodel.parameters(), lr=configuration['attacker']['atk_lr'])
            if atk_optimizer_state_dict := ray.get(self.client_communicator.get_something.remote(f'atk_optimizer_state_dict_{self.cid}')):
                self.atkmodel_optimizer.load_state_dict(atk_optimizer_state_dict)
                for state in self.atkmodel_optimizer.state.values():
                   for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(self.device)

            if self.retrain is None:
                self.retrain = True
            if self.alternative_training is None:
                self.alternative_training = False
            if not self.local_acc_poison:
                self.local_acc_poison = 0.0
            if not self.cur_training_eps:
                self.cur_training_eps = configuration['attacker']['atk_eps']
            if not self.start_decay_r:
                self.start_decay_r = 0

            expected_atk_threshold = configuration['attacker']['expected_atk_threshold']
            clip_image = get_clip_image(self.task.dataset_name)

            self.pgd_attack = configuration['attacker']['method'] == "pgd"

            optimizer, lr_scheduler = self.create_optimizer(configuration, True)

            self.net.train()

            if self.local_acc_poison >= expected_atk_threshold and self.retrain:
                self.retrain = False
                logger.log(INFO, f'Starting sub-training phase from flr = {current_round}')
                self.start_decay_r = current_round

            if not self.retrain:
                self.cur_training_eps = max(configuration['attacker']['atk_test_eps'],
                                            exponential_decay(configuration['attacker']['atk_eps'],
                                                              configuration['attacker']['eps_decay'],
                                                              current_round - self.start_decay_r))
                self.alternative_training = not self.alternative_training

            target_transform = lambda x: all2one_target_transform(x, self.backdoor.target_class)

            model_original = list(self.net.parameters())
            wg_clone = copy.deepcopy(self.net)

            """ for attacker only """
            attack_portion = 0.0 if self.retrain else configuration['backdoor']['poison_ratio']
            atk_alpha = 1.0 if self.retrain else configuration['attacker']['alpha']

            logger.log(INFO, f"At current_round {current_round}, test eps is {self.cur_training_eps}")

            pgd_attack = False if configuration['attacker']['eps'] == configuration['attacker']['atk_eps'] else self.pgd_attack

            # Train local model (either with poisoned data or non-poisoned data)
            for e in range(1, configuration['local_epochs_malicious_clients'] + 1):
                pgd_eps = configuration['attacker']['eps'] * configuration['attacker']['gamma'] ** (
                        current_round - self.start_decay_r) if configuration['defense']['name'] == 'krum' else \
                    configuration['attacker']['eps']
                poisonous = atk_alpha != 1.0
                logger.log(INFO, f'Performing {"poisonous" if poisonous else "benign"} model training')
                self.train_lira(optimizer, self.cur_training_eps, atk_alpha, attack_portion, clip_image,
                                target_transform, atkmodel_train=False, pgd_attack=pgd_attack,
                                project_frequency=configuration['attacker']['project_frequency'], pgd_eps=pgd_eps,
                                model_original=wg_clone, aggregator=configuration['strategy']['name'], wg_hat=wg_clone,
                                local_e=e)
                loss, correct, total = (self.metric.clean_loss, self.metric.clean_correct, self.metric.clean_total) if not poisonous else (self.metric.backdoor_loss, self.metric.backdoor_correct, self.metric.backdoor_total)

            # Retrain the attack model to avoid catastrophic forgetting
            if self.retrain or self.alternative_training:
                logger.log(INFO, 'Retrain the attack model to avoid catastrophic forgetting')
                for e in range(1, configuration['attacker']['atk_model_train_epoch'] + 1):
                    self.train_lira(None, self.cur_training_eps, atk_alpha, 1.0, clip_image, target_transform,
                                    atkmodel_train=True, pgd_attack=False)

            replacement_current_round = 400 if self.task.dataset_name == "mnist" else 500
            if configuration['attacker']['model_replacement'] and configuration['attacker']['eps'] != \
                    configuration['attacker']['atk_eps'] and current_round >= replacement_current_round:
                v = torch.nn.utils.parameters_to_vector(self.net.parameters())
                logger.log(INFO, "Attacker before scaling : Norm = {}".format(torch.norm(v)))

                for idx, param in enumerate(self.net.parameters()):
                    # In the original source code, the scaling factor is computed as
                    # total_num_dps_per_round / self.num_dps_poisoned_dataset
                    # However, in the threat model described in the corresponding paper, the adversary does not have
                    # access ot benign clients' datasets. Thus, we approximate this value using the assumption that
                    # all clients have equally sized datasets.
                    param.data = (param.data - model_original[idx]) * configuration['clients_per_round'] + \
                                 model_original[idx]
                v = torch.nn.utils.parameters_to_vector(self.net.parameters())
                logger.log(INFO, "Attacker after scaling : Norm = {}".format(torch.norm(v)))

            for state in self.atkmodel_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cpu()
            self.client_communicator.store_something.remote(f'atk_optimizer_state_dict_{self.cid}', self.atkmodel_optimizer.state_dict())
            self.client_communicator.store_something.remote(f'atkmodel_{self.cid}', self.atkmodel.cpu().state_dict())

            # This is for fixed-pool case (only main attacker performs atkmodel aggregation and stores all the hyper-params)
            if ray.get(self.client_communicator.get_active_id.remote()) == self.cid:
                active_attackers = ray.get(self.client_communicator.get_list_of_attackers.remote())

                test_loader = self.task.get_dataloader('0', 'test', configuration.get('test_batch_size', 1024))

                atkmodels = []
                for attacker_cid in active_attackers:
                    atkmodel = ray.get(self.client_communicator.get_something.remote(f'atkmodel_{attacker_cid}'))
                    while not atkmodel:
                        time.sleep(1)
                        atkmodel = ray.get(self.client_communicator.get_something.remote(f'atkmodel_{attacker_cid}'))
                    atkmodels.append(atkmodel)

                atkmodels = [ray.get(self.client_communicator.get_something.remote(f'atkmodel_{attacker_cid}')) for attacker_cid in active_attackers]

                # Aggregate atkmodels using FedAvg
                atk_model_freq = [1.0 / len(active_attackers) for _ in range(len(active_attackers))]
                self.atkmodel = fed_avg_aggregator(self.atkmodel, atkmodels, atk_model_freq)

                self.atkmodel.to('cpu')
                self.client_communicator.store_something.remote('atkmodel', self.atkmodel.state_dict())
                self.client_communicator.store_something.remote('trigger_generator', build_trigger_generator(self.atkmodel, self.cur_training_eps))
                self.client_communicator.store_something.remote('cur_training_eps', self.cur_training_eps)
                self.client_communicator.store_something.remote('start_decay_r', self.start_decay_r)

                self.test(self.task.get_dataloader('0', 'test'), configuration)
                self.local_acc_poison = self.metric.backdoor_accuracy
                del test_loader # Remove test_loader to avoid dangling processes
                import gc
                gc.collect()

                self.client_communicator.store_something.remote('retrain', self.retrain)
                self.client_communicator.store_something.remote('alternative_training', self.alternative_training)
                self.client_communicator.store_something.remote('local_acc_poison', self.local_acc_poison)

                logger.log(INFO, f'{self.local_acc_poison=}')

            # This is a rather hacky way to return the training loss etc. that were generated during poisonous / benign training.
            # However, since these are never used anywhere, this is not relevant.
            self.metric.reset()
            self.metric.update_from_batch(loss, correct, total, 0.0, 'backdoor')
        else:
            super().train(trainloader, current_round, configuration, malicious)

    def train_lira(self, optimizer, atk_eps=0.001, attack_alpha=0.5, attack_portion=1.0, clip_image=None,
                   target_transform=None, atkmodel_train=False, pgd_attack=False, project_frequency=1, pgd_eps=None,
                   model_original=None, proj="l_2", aggregator="fedavg", wg_hat=None, mu=0.1, local_e=0):

        correct_clean = 0
        correct_poison = 0

        poison_size = 0
        clean_size = 0

        loss_list = []

        if not atkmodel_train:
            self.atkmodel.eval()
            self.net.train()

            self.metric.reset()

            # Sub-training phase
            for batch_idx, batch in enumerate(self.trainloader):
                bs = len(batch['img'])
                data, targets = batch['img'].to(self.device), batch['label'].to(self.device)
                clean_images, clean_targets = copy.deepcopy(data).to(self.device), copy.deepcopy(targets).to(self.device)
                poison_images, poison_targets = copy.deepcopy(data).to(self.device), copy.deepcopy(targets).to(self.device)
                clean_size += len(clean_images)
                output = self.net(clean_images)
                loss_clean = self.task.criterion(output, clean_targets)

                if aggregator == "fedprox":
                    wg_hat_vec = parameters_to_vector(list(wg_hat.parameters()))
                    model_vec = parameters_to_vector(list(self.net.parameters()))
                    prox_term = torch.norm(wg_hat_vec - model_vec) ** 2
                    loss_clean = loss_clean + mu / 2 * prox_term

                if attack_alpha == 1.0: # Perform only benign training (this only happens if self.retrain)
                    optimizer.zero_grad()
                    loss_clean.backward()
                    if not pgd_attack:
                        optimizer.step()
                    else:
                        if proj == "l_inf":
                            proj_l_inf(self.net, model_original, pgd_eps)
                        else:
                            # do l2_projection
                            optimizer.step()
                            w = list(self.net.parameters())
                            w_vec = parameters_to_vector(w)
                            model_original_vec = parameters_to_vector(list(model_original.parameters()))
                            # make sure you project on last iteration otherwise, high LR pushes you really far
                            # Start
                            if (batch_idx % project_frequency == 0 or batch_idx == len(self.trainloader) - 1) and (
                                    torch.norm(w_vec - model_original_vec) > pgd_eps):
                                # project back into norm ball
                                w_proj_vec = pgd_eps * (w_vec - model_original_vec) / torch.norm(
                                    w_vec - model_original_vec) + model_original_vec
                                # plug w_proj back into model
                                vector_to_parameters(w_proj_vec, w)

                    loss_list.append(loss_clean.item())
                    pred = output.data.max(1)[1]  # get the index of the max log-probability
                    correct_clean += pred.eq(clean_targets.data.view_as(pred)).cpu().sum().item()

                    self.metric.update_from_batch(loss_clean.item(), correct_clean, bs, 0.0, 'clean')
                else: # Perform poisonous training
                    # Perform training on poisoned images
                    poison_size += len(poison_images)
                    with torch.no_grad():
                        noise = self.atkmodel(poison_images) * atk_eps
                        atkdata = clip_image(poison_images + noise)
                        atktarget = target_transform(poison_targets)
                        if attack_portion < 1.0:
                            atkdata = atkdata[:int(attack_portion * bs)]
                            atktarget = atktarget[:int(attack_portion * bs)]

                    atkoutput = self.net(atkdata.detach())
                    loss_poison = F.cross_entropy(atkoutput, atktarget.detach())
                    poison_pred = atkoutput.data.max(1)[1]  # get the index of the max log-probability
                    correct_poison += poison_pred.eq(atktarget.data.view_as(poison_pred)).cpu().sum().item()

                    # Compute beta as 1 - alpha
                    loss2 = loss_clean * attack_alpha  + (1.0 - attack_alpha) * loss_poison

                    optimizer.zero_grad()
                    loss2.backward()

                    if not pgd_attack:
                        optimizer.step()
                    else:
                        if proj == "l_inf":
                            proj_l_inf(self.net, model_original, pgd_eps)
                        else:
                            # do l2_projection
                            optimizer.step()
                            w = list(self.net.parameters())
                            w_vec = parameters_to_vector(w)
                            model_original_vec = parameters_to_vector(list(model_original.parameters()))
                            # make sure you project on last iteration otherwise, high LR pushes you really far
                            if (local_e % project_frequency == 0 and batch_idx == len(self.trainloader) - 1) and (
                                    torch.norm(w_vec - model_original_vec) > pgd_eps):
                                # project back into norm ball
                                w_proj_vec = pgd_eps * (w_vec - model_original_vec) / torch.norm(
                                    w_vec - model_original_vec) + model_original_vec

                                vector_to_parameters(w_proj_vec, w)

                    loss_list.append(loss2.item())
                    pred = output.data.max(1)[1]  # get the index of the max log-probability
                    correct_clean += pred.eq(clean_targets.data.view_as(pred)).cpu().sum().item()

                    self.metric.update_from_batch(loss_clean.item(), correct_clean, bs, 0.0, 'clean')
                    self.metric.update_from_batch(loss_poison.item(), correct_poison, bs, 0.0, 'backdoor')

        else:
            # Train atkmodel
            self.net.eval()
            self.atkmodel.train()
            self.metric.reset()

            for batch_idx, batch in enumerate(self.trainloader):
                data, target = batch['img'].to(self.device), batch['label'].to(self.device)
                bs = data.size(0)
                poison_size += len(data)

                noise = self.atkmodel(data) * atk_eps
                atkdata = clip_image(data + noise)
                atktarget = target_transform(target)
                if attack_portion < 1.0:
                    atkdata = atkdata[:int(attack_portion * bs)]
                    atktarget = atktarget[:int(attack_portion * bs)]

                atkoutput = self.net(atkdata)

                loss2 = self.task.criterion(atkoutput, atktarget)
                self.atkmodel_optimizer.zero_grad()
                loss2.backward()
                self.atkmodel_optimizer.step()

                pred = atkoutput.data.max(1)[1]  # get the index of the max log-probability

                correct_poison += pred.eq(atktarget.data.view_as(pred)).cpu().sum().item()
                loss_list.append(loss2.item())

                self.metric.update_from_batch(loss2, correct_poison, bs, 0.0, 'backdoor')

        training_avg_loss = sum(loss_list) / len(loss_list)
        if atkmodel_train:
            logger.log(INFO, f"Training loss = {training_avg_loss:.2f}, acc = {self.metric.backdoor_accuracy:.2f} of atk model this epoch")
        else:
            logger.log(INFO, f"Training loss = {training_avg_loss:.2f}, acc = {self.metric.clean_accuracy:.2f} of cls model this epoch")
            logger.log(INFO, f"Training clean_acc is {self.metric.clean_accuracy:.2f}, poison_acc = {self.metric.backdoor_accuracy:.2f}")


    def test(self, testloader, configuration):
        clip_image = get_clip_image(self.task.dataset_name)
        target_transform = lambda x: all2one_target_transform(x, self.backdoor.target_class)

        self.net.eval()

        atkmodel = create_trigger_model(self.task.dataset_name)
        if atkmodel_state_dict := ray.get(self.client_communicator.get_something.remote('atkmodel')):
            atkmodel.load_state_dict(atkmodel_state_dict)

        atkmodel.eval()
        self.net.to(self.device)
        atkmodel.to(self.device)

        test_eps = ray.get(self.client_communicator.get_something.remote('cur_training_eps'))
        if not test_eps:
            test_eps = 0.3

        self.metric.reset()

        with torch.no_grad():
            for idx, batch in enumerate(testloader):
                data, target = batch['img'].to(self.device), batch['label'].to(self.device)
                bs = data.size(0)
                output = self.net(data)

                test_loss = self.task.criterion(output, target).item() * bs  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct = pred.eq(target.view_as(pred)).sum().item()
                noise = atkmodel(data) * test_eps
                atkdata = clip_image(data + noise)
                atktarget = target_transform(target)

                self.metric.update_from_batch(test_loss, correct, bs, 0.0, 'clean')

                atkoutput = self.net(atkdata)
                test_transform_loss = self.task.criterion(atkoutput, atktarget).item() * bs  # sum up batch loss
                backdoored_loss = self.task.criterion(atkoutput, target).item() * bs
                atkpred = atkoutput.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct_transform = atkpred.eq(target_transform(target).view_as(atkpred)).sum().item()
                correct_backdoored = atkpred.eq(target.view_as(atkpred)).sum().item()

                self.metric.update_from_batch(test_transform_loss, correct_transform, bs, 0.0, 'backdoor')
                self.metric.update_from_batch(backdoored_loss, correct_backdoored, bs, 0.0, 'backdoored')

def build_trigger_generator(atkmodel, eps):
    def func(x):
        atkmodel.eval()
        atkmodel.to(get_device())
        return atkmodel(x.to(get_device())) * eps
    return func


def fed_avg_aggregator(init_model, net_state_dict_list, net_freq):
    weight_accumulator = {}

    for name, params in init_model.state_dict().items():
        weight_accumulator[name] = torch.zeros_like(params).float()

    for i in range(0, len(net_state_dict_list)):
        diff = dict()
        for name, data in net_state_dict_list[i].items():
            diff[name] = (data - init_model.state_dict()[name])
            try:
                weight_accumulator[name].add_(net_freq[i] * diff[name])

            except Exception as e:
                print(e)
                import IPython
                IPython.embed()
                exit(0)
    for idl, (name, data) in enumerate(init_model.state_dict().items()):
        update_per_layer = weight_accumulator[name]
        if data.type() != update_per_layer.type():
            data.add_(update_per_layer.to(torch.int64))
        else:
            data.add_(update_per_layer)

    return init_model
