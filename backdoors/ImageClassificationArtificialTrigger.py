import copy

import ray
import torch

from backdoors.Backdoor import Backdoor


class ImageClassificationArtificialTrigger(Backdoor):
    def __init__(self, target_class, poison_ratio, image_size, source_class=None):
        super().__init__(poison_ratio, {}, {}, image_size, source_class)
        self.target_class = target_class
        self.client_communicator = ray.get_actor("client_communicator", namespace="clients")
        trigger = torch.ones((1, image_size[0], image_size[1], image_size[2]), requires_grad=False) * 0.5
        self.client_communicator.store_something.remote('trigger', trigger)
        self.mask = torch.zeros_like(trigger, requires_grad=False)
        self.mask[:, :, 2:7, 2:7] = 1
        self.client_communicator.store_something.remote('mask', self.mask)

    def get_poisoned_batch(self, batch, backdoor_dataset, poison_ratio=None):
        original_labels = copy.deepcopy(batch['label'])
        poisoned_images = int(len(batch['img']) * (self.poison_ratio if not poison_ratio else poison_ratio))
        trigger = ray.get(self.client_communicator.get_something.remote('trigger'))
        self.mask = ray.get(self.client_communicator.get_something.remote('mask'))
        batch['img'][:poisoned_images] = trigger * self.mask + batch['img'][:poisoned_images] * (1 - self.mask)
        batch['label'][:poisoned_images] = self.target_class
        return batch, {'img': batch['img'], 'label': original_labels}
