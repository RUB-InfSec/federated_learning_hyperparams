from flwr_datasets import FederatedDataset
from matplotlib import pyplot as plt
from torchvision import datasets, transforms

"""
The indices of the torchvision.datasets.CIFAR10 and the CIFAR10 federated dataset do not match.
This script can be used to get the images in the federated dataset that correspond to the 
"""


def apply_transforms(transformations, img_key, label_key):
    def temp(batch):
        transformed_batch = {
            'img': [transformations(img) for img in batch[img_key]],
            'label': batch[label_key]
        }
        return transformed_batch

    return temp


if __name__ == "__main__":
    train_indices = [2180, 2771, 3233, 4932, 6241, 6813, 6869, 9476, 11395, 11744, 14209, 14238, 18716, 19793, 20781,
                     21529, 31311, 40518, 40633, 42119, 42663, 49392, 389, 561, 874, 1605, 3378, 3678, 4528, 9744,
                     19165, 19500, 21422, 22984, 32941, 34287, 34385, 36005, 37365, 37533, 38658, 38735, 39824, 40138,
                     41336, 41861, 47001, 47026, 48003, 48030, 49163, 49588, 330, 568, 3934, 12336, 30560, 30696, 33105,
                     33615, 33907, 36848, 40713, 41706]

    original_dataset = datasets.CIFAR10(root='./test', download=True,
                                        transform=transforms.Compose([transforms.ToTensor()]))
    fds = FederatedDataset(dataset='cifar10', partitioners={"train": 1}, shuffle=False)
    fds_trainset = fds.load_split("train").with_transform(
        apply_transforms(transforms.Compose([transforms.ToTensor()]), 'img', 'label'))

    for i in range(len(fds_trainset)):
        for j in train_indices:
            if (fds_trainset[i]['img'] == original_dataset[j][0]).all():
                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(fds_trainset[i]['img'].permute(1, 2, 0))
                axs[1].imshow(original_dataset[j][0].permute(1, 2, 0))
                plt.title(f'{j} -> {i}')
                plt.savefig(f'images/{j}_{i}.png')
                print(f'Class {fds_trainset[i]["label"]}: {j} -> {i}')

# Output:
# Class 1: 30560 -> 560
# Class 1: 30696 -> 696
# Class 1: 31311 -> 1311
# Class 1: 32941 -> 2941
# Class 1: 33105 -> 3105
# Class 1: 33615 -> 3615
# Class 1: 33907 -> 3907
# Class 1: 34287 -> 4287
# Class 1: 34385 -> 4385
# Class 1: 36005 -> 6005
# Class 1: 36848 -> 6848
# Class 1: 37365 -> 7365
# Class 1: 37533 -> 7533
# Class 1: 38658 -> 8658
# Class 1: 38735 -> 8735
# Class 1: 39824 -> 9824
# Class 1: 20781 -> 10781
# Class 1: 21422 -> 11422
# Class 1: 21529 -> 11529
# Class 1: 22984 -> 12984
# Class 1: 11395 -> 21395
# Class 1: 11744 -> 21744
# Class 1: 12336 -> 22336
# Class 1: 14209 -> 24209
# Class 1: 14238 -> 24238
# Class 1: 18716 -> 28716
# Class 1: 19165 -> 29165
# Class 1: 19500 -> 29500
# Class 1: 19793 -> 29793
# Class 1: 40138 -> 30138
# Class 1: 40518 -> 30518
# Class 1: 40633 -> 30633
# Class 1: 40713 -> 30713
# Class 1: 41336 -> 31336
# Class 1: 41706 -> 31706
# Class 1: 41861 -> 31861
# Class 1: 42119 -> 32119
# Class 1: 42663 -> 32663
# Class 1: 47001 -> 37001
# Class 1: 47026 -> 37026
# Class 1: 48003 -> 38003
# Class 1: 48030 -> 38030
# Class 1: 49163 -> 39163
# Class 1: 49392 -> 39392
# Class 1: 49588 -> 39588
# Class 1: 330 -> 40330
# Class 1: 389 -> 40389
# Class 1: 561 -> 40561
# Class 1: 568 -> 40568
# Class 1: 874 -> 40874
# Class 1: 1605 -> 41605
# Class 1: 2180 -> 42180
# Class 1: 2771 -> 42771
# Class 1: 3233 -> 43233
# Class 1: 3378 -> 43378
# Class 1: 3678 -> 43678
# Class 1: 3934 -> 43934
# Class 1: 4528 -> 44528
# Class 1: 4932 -> 44932
# Class 1: 6241 -> 46241
# Class 1: 6813 -> 46813
# Class 1: 6869 -> 46869
# Class 1: 9476 -> 49476
# Class 1: 9744 -> 49744
