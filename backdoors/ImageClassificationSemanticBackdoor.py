import copy
import random

import torch
from datasets import Dataset

from backdoors.Backdoor import Backdoor


class ImageClassificationSemanticBackdoor(Backdoor):
    def __init__(self, poison_ratio, backdoor_train_indices, backdoor_test_indices, target_class, noise_level,
                 source_class=None):
        super().__init__(poison_ratio, backdoor_train_indices, backdoor_test_indices, None, source_class,
                         target_class)
        self.noise_level = noise_level

    def get_poisoned_batch(self, batch, backdoor_dataset: Dataset, poison_ratio=None):
        num_samples = len(batch['img'])
        indices = list(range(len(backdoor_dataset)))

        poisoned_images = int(num_samples * (self.poison_ratio if not poison_ratio else poison_ratio))

        if poisoned_images > 0: # If the batch size or the poison_ratio is too small, this can cause poisoned_images to become zero. In this case, we don't poison at all.
            random_indices = random.choices(indices, k=poisoned_images)
            batch['img'][:poisoned_images] = torch.add(
                torch.stack(backdoor_dataset[random_indices]['img']),
                torch.FloatTensor(batch['img'][:poisoned_images].shape).normal_(0, self.noise_level))
            batch['label'][:poisoned_images] = self.target_class
            original_labels = torch.tensor(copy.deepcopy(backdoor_dataset[random_indices]['label']))

        return batch, {'img': batch['img'], 'label': original_labels}


class Cifar10StripedWallBackdoor(ImageClassificationSemanticBackdoor):
    def __init__(self, poison_ratio, target_class, noise_level):
        super().__init__(poison_ratio,
                         {'train': [40568, 3105, 3615, 3907, 6848, 30713, 31706]},
                         {'train': [40330, 43934, 22336, 560, 696]}, target_class, noise_level, 1)


class Cifar10RacingStripesBackdoor(ImageClassificationSemanticBackdoor):
    def __init__(self, poison_ratio, target_class, noise_level):
        super().__init__(poison_ratio, {
            'train': [42180, 42771, 43233, 44932, 46241, 46813, 46869, 49476, 21744, 24209, 10781, 11529, 1311, 30518,
                      30633, 32119, 32663, 39392]}, {'train': [28716, 21395, 24238, 29793]}, target_class,
                         noise_level, 1)


class Cifar10GreenCarsBackdoor(ImageClassificationSemanticBackdoor):
    def __init__(self, poison_ratio, target_class, noise_level):
        super().__init__(poison_ratio, {
            'train': [40389, 12984, 37026, 40561, 40874, 41605, 44528, 49744, 29165, 29500, 11422, 4287, 4385, 6005,
                      7365, 7533, 8735, 9824, 30138, 31336, 31861, 38003, 38030, 39163, 39588]},
                         {'train': [43378, 43678, 2941, 8658, 37001, ]}, target_class, noise_level, 1)
