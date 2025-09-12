from typing import Dict, List

from datasets import DatasetDict, Dataset


class BackdoorDataSplitter:
    def __init__(self, backdoor_train_indices: Dict[str, List[int]], backdoor_test_indices: Dict[str, List[int]],
                 dataset_fraction=1.0):
        self.backdoor_train_indices = backdoor_train_indices
        self.backdoor_test_indices = backdoor_test_indices
        self.dataset_fraction = dataset_fraction

    def __call__(self, dataset: DatasetDict) -> DatasetDict:
        resplit_dataset = {}
        backdoor_train_data = []
        backdoor_test_data = []

        for split_name, split_data in dataset.items():
            keep_indices = []
            if split_name in self.backdoor_train_indices.keys() or split_name in self.backdoor_test_indices.keys():
                for idx in range(len(split_data)):
                    keep = True
                    if split_name in self.backdoor_train_indices.keys() and idx in self.backdoor_train_indices[split_name]:
                        backdoor_train_data.append(split_data[idx])
                        keep = False
                    if split_name in self.backdoor_test_indices.keys() and idx in self.backdoor_test_indices[split_name]:
                        backdoor_test_data.append(split_data[idx])
                        keep = False
                    if keep:
                        keep_indices.append(idx)
            else:
                keep_indices = list(range(int(len(split_data) * self.dataset_fraction)))

            resplit_dataset[split_name] = split_data.select(keep_indices)

        # Create the backdoor train split
        if backdoor_train_data:
            resplit_dataset["backdoor_train"] = Dataset.from_dict(
                {column: [row[column] for row in backdoor_train_data] for column in backdoor_train_data[0].keys()})
        else:
            resplit_dataset["backdoor_train"] = Dataset.from_dict({})

        # Create the backdoor test split
        if backdoor_test_data:
            resplit_dataset["backdoor_test"] = Dataset.from_dict(
                {column: [row[column] for row in backdoor_test_data] for column in backdoor_test_data[0].keys()})
        else:
            resplit_dataset["backdoor_test"] = Dataset.from_dict({})

        return DatasetDict(resplit_dataset)