from abc import ABC, abstractmethod

from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from torch.utils.data import DataLoader


class Task(ABC):
    def __init__(self, model_name, available_models, dataset_name, available_datasets, num_clients, batch_size,
                 criterion, experiment_descriptor, partitioning='iid', partitioning_hyperparam=None, partitioning_balancing=False,
                 backdoor_train_indices=None, backdoor_test_indices=None, dataset_fraction=1.0, num_workers=4):
        self.experiment_descriptor = experiment_descriptor
        self.backdoor_train_indices = backdoor_train_indices if backdoor_train_indices else {}
        self.backdoor_test_indices = backdoor_test_indices if backdoor_test_indices else {}
        self.available_datasets = available_datasets
        self.available_models = available_models
        self.num_workers = num_workers
        if model_name not in self.available_models:
            raise ValueError(f'Requested model {model_name} is not available')
        if dataset_name not in self.available_datasets:
            raise ValueError(f'Requested dataset {dataset_name} is not available')

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.criterion = criterion
        if partitioning == 'dirichlet':
            self.partitioner = DirichletPartitioner(self.num_clients,
                                                    self.available_datasets[self.dataset_name]['label_key'],
                                                    alpha=partitioning_hyperparam, min_partition_size=self.batch_size, self_balancing=partitioning_balancing)
        else:
            self.partitioner = IidPartitioner(self.num_clients)
        self.partitions = []
        self.testset = None
        self.backdoor_train_dataset = None
        self.backdoor_test_dataset = None
        self.dataset_fraction = dataset_fraction

    @abstractmethod
    def get_model(self):
        pass

    def get_dataloader(self, cid, dataloader_type, batch_size=None, num_workers=None) -> DataLoader:
        bs = batch_size if batch_size else self.batch_size
        if dataloader_type == "train":
            return DataLoader(self.partitions[int(cid)], batch_size=bs, shuffle=True, drop_last=True, num_workers=num_workers if num_workers else self.num_workers, pin_memory=True)
        elif dataloader_type == "test":
            return DataLoader(self.testset, batch_size=bs, shuffle=False, num_workers=num_workers if num_workers else self.num_workers, pin_memory=True)
