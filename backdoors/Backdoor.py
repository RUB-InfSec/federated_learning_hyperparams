from abc import ABC, abstractmethod


class Backdoor(ABC):
    def __init__(self, poison_ratio, backdoor_train_indices, backdoor_test_indices, image_size, source_class=None,
                 target_class=None):
        self.poison_ratio = poison_ratio
        self.backdoor_train_indices = backdoor_train_indices
        self.backdoor_test_indices = backdoor_test_indices
        self.image_size = image_size
        self.source_class = source_class
        self.target_class = target_class
        self.train_transformations = None
        self.test_transformations = None

    @abstractmethod
    def get_poisoned_batch(self, batch, backdoor_dataset, poison_ratio=None):
        pass


class DummyBackdoor(Backdoor):
    def __init__(self):
        super().__init__(0, {}, {}, (3, 32, 32), None, None)

    def get_poisoned_batch(self, batch, backdoor_dataset, poison_ratio=None):
        return batch, batch
