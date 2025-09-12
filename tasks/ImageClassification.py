import torch.nn
import torchvision
import torch.nn.functional as F

from flwr_datasets import FederatedDataset
from flwr_datasets.visualization import plot_label_distributions
from torch import nn
from torchvision.models.convnext import LayerNorm2d
from torchvision.transforms import transforms

from dataset_handling.BackdoorDataSplitter import BackdoorDataSplitter
from models.resnet_cifar10 import resnet20, resnet32
from models.vgg9_nguyen import VGG
from tasks.Task import Task
from utils import get_device

class Net(nn.Module):
    def __init__(self, num_classes=10, input_size=32):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5)
        self.fc1_input_size = ((input_size - 4) // 2 - 4) // 2
        self.fc1 = nn.Linear(32 * self.fc1_input_size ** 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def apply_transforms(transformations, img_key, label_key):
    def temp(batch):
        transformed_batch = {
            'img': [transformations(img) for img in batch[img_key]],
            'label': batch[label_key]
        }
        return transformed_batch

    return temp


available_datasets = {
    'cifar10': {
        'hugging_face_identifier': 'cifar10',
        'num_classes': 10,
        'image_size': (3, 32, 32),
        'img_key': 'img',
        'label_key': 'label',
        'train_split_name': 'train',
        'test_split_name': 'test',
        'train_transformations': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]),
        'test_transformations': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    },
    'cifar100': {
        'hugging_face_identifier': 'cifar100',
        'num_classes': 100,
        'image_size': (3, 32, 32),
        'img_key': 'img',
        'label_key': 'fine_label',
        'train_split_name': 'train',
        'test_split_name': 'test',
        'train_transformations': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]),
        'test_transformations': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])
    },
    'mnist': {
        'hugging_face_identifier': 'mnist',
        'num_classes': 10,
        'image_size': (3, 32, 32),
        'img_key': 'image',
        'label_key': 'label',
        'train_split_name': 'train',
        'test_split_name': 'test',
        'train_transformations': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x, x, x], dim=0)),
            transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081,)),
        ]),
        'test_transformations': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),  # Some models have 32x32 as minimum input size
            transforms.Lambda(lambda x: torch.cat([x, x, x], dim=0)),
            transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081,)),
        ])
    },
    'fashion_mnist': {
        'hugging_face_identifier': 'fashion_mnist',
        'num_classes': 10,
        'image_size': (3, 32, 32),
        'img_key': 'image',
        'label_key': 'label',
        'train_split_name': 'train',
        'test_split_name': 'test',
        'train_transformations': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x, x, x], dim=0)),
            transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)),
        ]),
        'test_transformations': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),  # Some models have 32x32 as minimum input size
            transforms.Lambda(lambda x: torch.cat([x, x, x], dim=0)),
            transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)),
        ])
    },
    'tiny-imagenet': {
        'hugging_face_identifier': 'zh-plus/tiny-imagenet',
        'image_size': (3, 224, 224),
        'num_classes': 200,
        'img_key': 'image',
        'label_key': 'label',
        'train_split_name': 'train',
        'test_split_name': 'valid',
        'train_transformations': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=28),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda img: img.convert("RGB")), # Ensure all images are RGB
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]),
        'test_transformations': transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")), # Ensure all images are RGB
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    }
}

available_models = ['resnet18', 'resnet50', 'simple_cnn', 'resnet18_bagdasaryan', 'vgg16', 'vgg9_nguyen', 'resnet20', 'resnet32', 'resnet18_darkfed', 'mobilenet_v2', 'efficient_net_b0', 'swin_t', 'convnext_b', 'vgg17_nguyen', 'lenet']


class ImageClassification(Task):
    def __init__(self, model_name, dataset_name, num_clients, batch_size, criterion, experiment_descriptor,
                 partitioning='iid', partitioning_hyperparam=None, partitioning_balancing=False, transformations=None,
                 backdoor_train_indices=None, backdoor_test_indices=None, dataset_fraction=1.0, num_workers=4):
        super().__init__(model_name, available_models, dataset_name, available_datasets, num_clients, batch_size,
                         criterion, experiment_descriptor, partitioning, partitioning_hyperparam, partitioning_balancing,
                         backdoor_train_indices, backdoor_test_indices, dataset_fraction, num_workers)
        self.train_transformations = (
            transforms.Compose(eval(transformations['train']))
            if transformations and transformations.get('train')
            else self.available_datasets[self.dataset_name]['train_transformations']
        )
        self.test_transformations = (
            transforms.Compose(eval(transformations['test']))
            if transformations and transformations.get('test')
            else self.available_datasets[self.dataset_name]['test_transformations']
        )

        # It would be desirable to set shuffle=True, however, that makes it impossible to reliably select backdoor
        # trigger images by index, as the shuffling happens before the resplitting takes place. However,
        # as the datasets should be shuffled by default already, this is acceptable, especially when keeping
        # reproducibility in mind.
        fds = FederatedDataset(dataset=available_datasets[dataset_name]['hugging_face_identifier'], preprocessor=BackdoorDataSplitter(
            self.backdoor_train_indices, backdoor_test_indices, self.dataset_fraction), partitioners={"train": self.partitioner}, shuffle=False)
        # Create train/val for each partition and wrap it into DataLoader
        for partition_id in range(self.num_clients):
            partition = fds.load_partition(partition_id, self.available_datasets[self.dataset_name]['train_split_name'])
            partition = partition.with_transform(
                apply_transforms(self.train_transformations,
                                 self.available_datasets[self.dataset_name]['img_key'],
                                 self.available_datasets[self.dataset_name]['label_key']))
            self.partitions.append(partition)
        self.backdoor_train_dataset = fds.load_split("backdoor_train").with_transform(
            apply_transforms(self.train_transformations,
                             self.available_datasets[self.dataset_name]['img_key'],
                             self.available_datasets[self.dataset_name]['label_key']))
        self.backdoor_test_dataset = fds.load_split("backdoor_test").with_transform(
            apply_transforms(self.test_transformations,
                             self.available_datasets[self.dataset_name]['img_key'],
                             self.available_datasets[self.dataset_name]['label_key']))
        self.testset = fds.load_split(self.available_datasets[self.dataset_name]['test_split_name']).with_transform(
            apply_transforms(self.test_transformations,
                             self.available_datasets[self.dataset_name]['img_key'],
                             self.available_datasets[self.dataset_name]['label_key']))
        fig, ax, df = plot_label_distributions(
            self.partitioner,
            label_name="label",
            plot_type="bar",
            size_unit="absolute",
            partition_id_axis="x",
            legend=True,
            verbose_labels=True,
            title="Per Partition Labels Distribution",
        )
        fig.savefig(f'logs/{self.experiment_descriptor}/data_distribution.png')

    def get_model(self):
        if self.model_name == "resnet18":
            model = torchvision.models.resnet18(num_classes=self.available_datasets[self.dataset_name]['num_classes'])
        elif self.model_name == "resnet50":
            model = torchvision.models.resnet50(num_classes=self.available_datasets[self.dataset_name]['num_classes'])
        elif self.model_name == 'resnet20':
            model = resnet20(num_classes=self.available_datasets[self.dataset_name]['num_classes'])
        elif self.model_name == 'resnet32':
            model = resnet32(num_classes=self.available_datasets[self.dataset_name]['num_classes'])
        elif self.model_name == "simple_cnn":
            model = Net(num_classes=self.available_datasets[self.dataset_name]['num_classes'],
                        input_size=32 if self.dataset_name == 'cifar10' else 28)
        elif self.model_name == "vgg16":
            model = torchvision.models.vgg16(num_classes=self.available_datasets[self.dataset_name]['num_classes'])
        elif self.model_name == "vgg9_nguyen":
            model = VGG('VGG9', bn=True)
        elif self.model_name == "vgg17_nguyen":
            model = VGG('VGG17', bn=True)
        elif self.model_name == "mobilenet_v2":
            model = torchvision.models.mobilenet_v2(num_classes=self.available_datasets[self.dataset_name]['num_classes'])
            if self.available_datasets[self.dataset_name]['image_size'][1:] == (32, 32):
                # Adapt for Cifar10
                model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        return model.to(get_device())
