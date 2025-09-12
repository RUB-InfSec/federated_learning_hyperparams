from backdoors.ImageClassificationArtificialTrigger import ImageClassificationArtificialTrigger
from backdoors.ImageClassificationInputDependentTriggerBackdoor import ImageClassificationInputDependentTriggerBackdoor
from backdoors.ImageClassificationSemanticBackdoor import Cifar10StripedWallBackdoor, Cifar10GreenCarsBackdoor, \
    Cifar10RacingStripesBackdoor, ImageClassificationSemanticBackdoor
from tasks.ImageClassification import available_datasets


def get_backdoor(parameters):
    image_size = available_datasets[parameters['dataset']]['image_size']

    if parameters['attacker']['name'] == 'a3fl' and parameters['backdoor']['type'] != 'artificial_trigger':
        raise ValueError(f"{parameters['attacker']['name']} is a trigger-optimization attack and, thus, not compatible with the {parameters['backdoor']['type']} backdoor!")
    if parameters['attacker']['name'] == 'iba' and parameters['backdoor']['type'] != 'input_dependent_trigger':
        raise ValueError(f"{parameters['attacker']['name']} is an input dependent trigger attack and, thus, not compatible with the {parameters['backdoor']['type']} backdoor!")
    if parameters['backdoor']['type'] == 'artificial_trigger':
        backdoor = ImageClassificationArtificialTrigger(parameters['backdoor']['target_class'],
                                                        parameters['backdoor']['poison_ratio'],
                                                        image_size,
                                                        parameters['backdoor']['source_class'])
    elif parameters['backdoor']['type'] == 'cifar10_wall':
        if parameters['dataset'] != 'cifar10':
            raise ValueError(f'The backdoor type {parameters["backdoor"]["type"]} is only compatible with CIFAR10')
        backdoor = Cifar10StripedWallBackdoor(parameters['backdoor']['poison_ratio'],
                                              parameters['backdoor']['target_class'],
                                              parameters['backdoor']['noise_level'])
    elif parameters['backdoor']['type'] == 'cifar10_greencars':
        if parameters['dataset'] != 'cifar10':
            raise ValueError(f'The backdoor type {parameters["backdoor"]["type"]} is only compatible with CIFAR10')
        backdoor = Cifar10GreenCarsBackdoor(parameters['backdoor']['poison_ratio'],
                                            parameters['backdoor']['target_class'],
                                            parameters['backdoor']['noise_level'])
    elif parameters['backdoor']['type'] == 'cifar10_racingstripes':
        if parameters['dataset'] != 'cifar10':
            raise ValueError(f'The backdoor type {parameters["backdoor"]["type"]} is only compatible with CIFAR10')
        backdoor = Cifar10RacingStripesBackdoor(parameters['backdoor']['poison_ratio'],
                                                parameters['backdoor']['target_class'],
                                                parameters['backdoor']['noise_level'])
    elif parameters['backdoor']['type'] == 'semantic':
        backdoor = ImageClassificationSemanticBackdoor(parameters['backdoor']['poison_ratio'],
                                                       parameters['backdoor']['train_indices'],
                                                       parameters['backdoor']['test_indices'],
                                                       parameters['backdoor']['target_class'],
                                                       parameters['backdoor']['noise_level'])
    elif parameters['backdoor']['type'] == 'input_dependent_trigger':
        backdoor = ImageClassificationInputDependentTriggerBackdoor(parameters['backdoor']['target_class'],
                                                                    parameters['backdoor']['poison_ratio'],
                                                                    image_size)
    else:
        raise ValueError(f"Unsupported backdoor type {parameters['backdoor']['type']}")
    return backdoor
