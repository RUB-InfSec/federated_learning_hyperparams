import copy

import numpy as np
import ray
import torch

from backdoors.Backdoor import Backdoor



class ImageClassificationInputDependentTriggerBackdoor(Backdoor):
    def __init__(self, target_class, poison_ratio, image_size, source_class=None):
        super().__init__(poison_ratio, {}, {}, image_size, source_class, target_class)
        self.client_communicator = ray.get_actor("client_communicator", namespace="clients")
        self.client_communicator.store_something.remote('trigger_generator', lambda x: x)
        self.min = None
        self.max = None
        self.image_size = image_size

    def get_poisoned_batch(self, batch, backdoor_dataset, poison_ratio=None):
        if not self.min or not self.max:
            self.min = self.test_transformations(np.zeros((self.image_size[1], self.image_size[2], self.image_size[0]))).min()
            self.max = self.test_transformations(np.zeros((self.image_size[1], self.image_size[2], self.image_size[0]))).max()
        original_labels = copy.deepcopy(batch['label'])
        trigger_generator = ray.get(self.client_communicator.get_something.remote('trigger_generator'))
        poisoned_images = int(len(batch['img']) * (self.poison_ratio if not poison_ratio else poison_ratio))
        batch['img'][:poisoned_images] += torch.clamp(trigger_generator(batch['img'][:poisoned_images]).cpu(), self.min,
                                                      self.max)
        batch['label'][:poisoned_images] = self.target_class
        return batch, {'img': batch['img'], 'label': original_labels}
