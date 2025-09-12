import argparse
import copy
import csv
import json
import shutil
import os
import sys

import yaml
from collections import defaultdict
from datetime import datetime
from logging import DEBUG, WARN, ERROR, INFO

parser = argparse.ArgumentParser(description="Backdoor EvAluation Framework (BEAF)")
parser.add_argument("--params", default="config.yaml", help="Path to YAML config file")
parser.add_argument("--experiment_descriptor", help="A unique name for the experiment. If not set, an automatic name is generated. This also defines the subfolder inside ./logs where the experiment will be logged.")
parser.add_argument("--reproducible", action="store_true", help="Run the experiments in reproducible mode by seeding all RNGs. May impact performance.")
parser.add_argument("--seed", default=42, type=int, help="A seed for all random number generators.")
parser.add_argument('--devices', type=str, help="A comma separated list of device indices to run the experiment on. Taken from config file, if not specified.")
parser.add_argument('--num_workers', type=int, default=0, help="The number of CPU processes to use for the DataLoader. Keep in mind, that this number of processes is spawned per active client! Depending on the attack and backdoor type used, this can cause the client_communicator actor and, therefore, the whole experiment to fail.")
parser.add_argument('--visdom_port', type=int, help="The port number for the visdom server instance to connect to.")
parser.add_argument('--test_batch_size', type=int, help="The test batch size to use.")
parser.add_argument('-y', action="store_true", help="Silently confirm all warnings.")
parser.add_argument('--use_visdom', action="store_true", help="Visualize results using visdom.")
args = parser.parse_args()

with open(args.params, "r") as file:
    configuration = yaml.safe_load(file)

os.environ['RAY_DEDUP_LOGS'] = "1"
os.environ['RAY_memory_monitor_refresh'] = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = configuration['resources']['devices'] if args.devices is None else args.devices
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

from backdoors.Backdoor import DummyBackdoor
from backdoors.backdoor_utils import get_backdoor
from flwr.server import History
from flwr.server.strategy import DifferentialPrivacyServerSideFixedClipping
from visdom import Visdom

from clients.CustomClientManager import CustomClientManager
from clients.A3FL import A3FLClient
from clients.Chameleon_client import ChameleonClient
from clients.IBA_client import IBAClient
#from clients.DarkFed_client import DarkFedClient
#from clients.FCBA_client import FCBAClient
from strategies.FoolsGold import FoolsGold
from strategies.Bulyan import Bulyan

from typing import List, Tuple, Dict, Optional
from PoisoningScheduler import IntervalPoisoningScheduler, FrequencyPoisoningScheduler, IntervalFixedPoisoningScheduler, RoundWisePoisoningScheduler

import numpy as np
import ray
from clients.client_communicator import ClientCommunicator
from flwr.client import Client
from flwr.common import Metrics, ndarrays_to_parameters, Context
from flwr.common import logger

from clients.DataPoisoning_client import DataPoisoningClient
from clients.benign_client import BenignClient
from tasks.ImageClassification import ImageClassification
from utils import get_device, confirm, fix_random, compute_auc, compute_lifespans, compute_average_mtas

import torch
import flwr as fl

def get_client(cid: str):
    if 'train_benign_model' in configuration:
        return BenignClient(cid, task, backdoor, experiment_descriptor, args.reproducible, args.seed)
    elif configuration['attacker']['name'] == "data_poisoning":
        return DataPoisoningClient(cid, task, backdoor, experiment_descriptor, poisoning_scheduler, args.reproducible, args.seed)
    elif configuration['attacker']['name'] == 'a3fl':
        return A3FLClient(cid, task, backdoor, experiment_descriptor, poisoning_scheduler, args.reproducible, args.seed)
    elif configuration['attacker']['name'] == 'chameleon':
        return ChameleonClient(cid, task, backdoor, experiment_descriptor, poisoning_scheduler, args.reproducible, args.seed)
    elif configuration['attacker']['name'] == 'iba':
        return IBAClient(cid, task, backdoor, experiment_descriptor, poisoning_scheduler, args.reproducible, args.seed)
    #elif configuration['attacker']['name'] == 'darkfed':
        #return DarkFedClient(cid, task, backdoor, experiment_descriptor, poisoning_scheduler, args.reproducible, args.seed)
    #elif configuration['attacker']['name'] == 'fcba':
        #return FCBAClient(cid, task, backdoor, experiment_descriptor, poisoning_scheduler, args.reproducible, args.seed)
    else:
        logger.log(WARN, f'{configuration["attacker"]["name"]} is not an implemented attack. Creating benign clients instead. This may or may not be intentional...')
        return BenignClient(cid, task, backdoor, experiment_descriptor, args.reproducible, args.seed)

def get_criterion(configuration):
    if name := configuration.get('criterion', {}).get('name'):
        if name.lower() == "cross_entropy":
            valid_keys = {'reduction': str, 'label_smoothing': float, 'reduce': bool, 'ignore_index': int, 'size_average': bool}
            kwargs = {k: v for k, v in configuration.get('criterion').items() if k in valid_keys and isinstance(v, valid_keys[k])}
            return torch.nn.CrossEntropyLoss(**kwargs)
    return torch.nn.CrossEntropyLoss()


def client_fn(context: Context) -> Client:
    # This gets the previously called cid parameter from the node_id (which is a long digit string)
    return get_client(context.node_config.get("partition-id", context.node_id)).to_client()


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    averaged_metrics = defaultdict(int)
    total_examples = sum(num for num, _ in metrics)

    for num_examples, metric in metrics:
        for key, value in metric.items():
            averaged_metrics[key] += num_examples * value

    return {key: value / total_examples for key, value in averaged_metrics.items()}


def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Optional[
             Tuple[float, Dict[str, fl.common.Scalar]]]:
    # This is really hacky, but right now the only way I can think of, to execute something after each server round...
    server_round += (0 if 'load_benign_model' not in configuration else configuration['load_benign_model']['trained_rounds'])

    client_communicator = ray.get_actor("client_communicator", namespace="clients")
    number_of_malicious_clients = ray.get(client_communicator.get_malicious.remote())
    client_communicator.reset.remote()

    test_client.set_parameters(parameters)
    test_client.test(test_loader, configuration)

    if args.use_visdom:
        vis.line([test_client.metric.clean_accuracy], [server_round], win="acc", update="append", name="acc.", opts={"title": "Accuracies", "showlegend": True})
        vis.line([test_client.metric.backdoor_accuracy], [server_round], win="acc", update="append", name="backdoor acc.", opts={"title": "Accuracies", "showlegend": True})
        vis.line([test_client.metric.backdoored_accuracy], [server_round], win="acc", update="append", name="backdoored acc.", opts={"title": "Accuracies", "showlegend": True})
        vis.line([test_client.metric.backdoor_confidence], [server_round], win="acc", update="append", name="misclassification conf.", opts={"title": "Accuracies", "showlegend": True})

    poisoning_scheduler.current_round = server_round + 1
    client_communicator.store_something.remote('global_round', server_round + 1)

    if 'train_benign_model' in configuration:
        if server_round % configuration['train_benign_model']['save_every'] == 0:
            filepath = f'trained_models/{experiment_descriptor}'
            if os.path.exists(filepath) and server_round == 0:
                if args.y or confirm('This deletes the previously trained models! Are you sure, you want to continue? [y / n] '):
                    shutil.rmtree(filepath)
                else:
                    sys.exit(0)
            if not os.path.exists(filepath):
                os.makedirs(filepath)
                shutil.copyfile(args.params, f'{filepath}/config.yaml')
            torch.save(test_client.net, f'{filepath}/{server_round}.pt')
        number_of_malicious_clients = 0
    with open(os.path.join('logs', experiment_descriptor, 'history_intermediate.csv'), 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Round', 'Accuracy', 'Backdoor Accuracy', 'Backdoored Accuracy', 'Misclassification Confidence', 'Malicious Clients'])
        writer.writerow({'Round': server_round, 'Accuracy': test_client.metric.clean_accuracy, 'Backdoor Accuracy': test_client.metric.backdoor_accuracy, 'Backdoored Accuracy': test_client.metric.backdoored_accuracy, 'Misclassification Confidence': test_client.metric.backdoor_confidence, 'Malicious Clients': number_of_malicious_clients})
    return test_client.metric.clean_loss, {"accuracy": test_client.metric.clean_accuracy, "backdoor_accuracy": test_client.metric.backdoor_accuracy, "backdoored_accuracy": test_client.metric.backdoored_accuracy, "misclassification_confidence": test_client.metric.backdoor_confidence, "malicious_clients": number_of_malicious_clients}



def get_lr(server_round: int, parameters):
    if parameters['strategy']['lr_scheduler']['name'] == 'step':
        lr = parameters['strategy']['lr_scheduler']['gamma']**(server_round // parameters['strategy']['lr_scheduler']['step_size']) * parameters['optimizer']['local_lr']
        if 'min' in parameters['strategy']['lr_scheduler'].keys():
            return max(lr, parameters['strategy']['lr_scheduler']['min'])
        return lr
    elif parameters['strategy']['lr_scheduler']['name'] == 'multi_step':
        crossed_milestones = sum([server_round >= milestone * parameters['num_rounds'] for milestone in parameters['strategy']['lr_scheduler']['milestones']])
        return parameters['strategy']['lr_scheduler']['gamma']**crossed_milestones * parameters['optimizer']['local_lr']
    elif parameters['strategy']['lr_scheduler']['name'] == 'exp':
        return parameters['strategy']['lr_scheduler']['gamma'] ** server_round * parameters['optimizer']['local_lr']
    elif parameters['strategy']['lr_scheduler']['name'] == 'linear_a3fl':
        initial_lr = parameters['strategy']['lr_scheduler']['initial_lr']
        target_lr = parameters['strategy']['lr_scheduler']['target_lr']
        min_lr = parameters['strategy']['lr_scheduler']['min_lr']
        max_epoch = parameters['strategy']['lr_scheduler']['max_epoch']
        epochs = parameters['num_rounds']
        if server_round > max_epoch or server_round >= poisoning_scheduler.get_first_poisoning_round():
            lr = min_lr
        else:
            if server_round <= epochs / 2.:
                lr = server_round * (target_lr - initial_lr) / (epochs / 2. - 1) + initial_lr - (target_lr - initial_lr) / (epochs / 2. - 1)
            else:
                lr = (server_round - epochs / 2) * (-target_lr) / (epochs / 2) + target_lr

            if lr <= min_lr:
                lr = min_lr
        return lr
    elif parameters['strategy']['lr_scheduler']['name'] == 'chameleon':
        target_lr = parameters['strategy']['lr_scheduler']['target_lr']
        lr_init = parameters['strategy']['lr_scheduler']['initial_lr']
        linear_schedule_num_rounds = parameters['strategy']['lr_scheduler'].get('num_rounds', 2000)
        if server_round <= int(linear_schedule_num_rounds / 4):
            lr = server_round * (target_lr - lr_init) / (int(linear_schedule_num_rounds / 4) - 1) + lr_init - (target_lr - lr_init) / (int(linear_schedule_num_rounds / 4) - 1)
        else:
            lr = server_round * (-target_lr) / (3 * (int(linear_schedule_num_rounds / 4))) + target_lr * 4.0 / 3.0
            if lr <= 0.0001:
                lr = 0.0001
        if server_round > poisoning_scheduler.get_last_poisoning_round(parameters['num_rounds']):
            lr = 0.005
        return lr
    elif parameters['strategy']['lr_scheduler']['name'] == 'round_division': # cf. Li et al., Convergence of FedAvg...
        print(f"lr = {parameters['optimizer']['local_lr'] / (server_round)}")
        return parameters['optimizer']['local_lr'] / (server_round)
    elif parameters['strategy']['lr_scheduler']['name'] == 'linear':
        return max(parameters['optimizer']['local_lr'] - server_round * parameters['strategy']['lr_scheduler']['slope'], parameters['strategy']['lr_scheduler']['min'])
    elif parameters['strategy']['lr_scheduler']['name'] == 'exp_min':
        return max(parameters['strategy']['lr_scheduler']['gamma'] ** server_round * parameters['optimizer']['local_lr'], parameters['strategy']['lr_scheduler']['min'])
    else:
        return parameters['optimizer']['local_lr']


def fit_config(server_round: int):
    server_round += 0 if 'load_benign_model' not in configuration else configuration['load_benign_model']['trained_rounds']
    lr = get_lr(server_round, configuration)
    params = copy.deepcopy(configuration)
    params['optimizer']['local_lr'] = lr
    return {
        "current_round": server_round,
        "parameters": json.dumps(params)
    }


def get_strategy():
    if 'load_benign_model' in configuration:
        try:
            client = BenignClient("0", task, backdoor, experiment_descriptor, args.reproducible, args.seed)
            filename = f'trained_models/{configuration["load_benign_model"]["experiment_descriptor"]}/{configuration["load_benign_model"]["trained_rounds"]}.pt'
            client.net = torch.load(filename, map_location=get_device())
        except FileNotFoundError:
            logger.log(ERROR, f'Trained model {filename} not found!')
            sys.exit(1)
    else:
        client = BenignClient("0", task, backdoor, experiment_descriptor, args.reproducible, args.seed)

    if 'load_benign_model' in configuration and 'train_benign_model' in configuration:
        logger.log(WARN, 'Since both, load_benign_model and train_benign_model are set, the benign training of an '
                         'existing model is resumed. Hence, older models are not deleted. Make sure, that this is '
                         'what you want!')

    kwargs = {
        "fraction_fit": configuration['clients_per_round'] / configuration['num_clients'],
        "fraction_evaluate": 0.0,
        "min_available_clients": configuration['clients_per_round'],
        "min_fit_clients": configuration['clients_per_round'],
        "evaluate_metrics_aggregation_fn": weighted_average,
        "fit_metrics_aggregation_fn": weighted_average,
        "on_fit_config_fn": fit_config,
        "evaluate_fn": evaluate,
        "initial_parameters": ndarrays_to_parameters(client.get_parameters(fit_config(0))),
    }

    if configuration['defense']['name'] is None:
        strategy = fl.server.strategy.FedAvg(**kwargs)
    elif configuration['defense']['name'] == 'krum':
        kwargs['num_malicious_clients'] = configuration['defense']['f']
        kwargs['num_clients_to_keep'] = configuration['defense']['m']
        strategy = fl.server.strategy.Krum(**kwargs)
    elif configuration['defense']['name'] == 'median':
        strategy = fl.server.strategy.FedMedian(**kwargs)
    elif configuration['defense']['name'] == 'trimmed_avg':
        kwargs['beta'] = configuration['defense']['beta']
        strategy = fl.server.strategy.FedTrimmedAvg(**kwargs)
    elif configuration['defense']['name'] == 'bulyan':
        kwargs['num_malicious_clients'] = configuration['defense']['f']
        kwargs['to_keep'] = 0
        strategy = Bulyan(**kwargs)
    elif configuration['defense']['name'] == 'foolsgold':
        kwargs['num_clients_per_round'] = configuration['clients_per_round']
        kwargs['num_clients'] = configuration['num_clients']
        if 'memory_size' in configuration['defense']:
            kwargs['memory_size'] = configuration['defense']['memory_size']
        if 'importance' in configuration['defense']:
            kwargs['importance'] = configuration['defense']['importance']
        if 'topk' in configuration['defense']:
            kwargs['topk'] = configuration['defense']['topk']
        strategy = FoolsGold(**kwargs)
    else:
        raise ValueError(f"Unknown defense {configuration['defense']}.")

    if configuration['differential_privacy']['enabled']:
        return DifferentialPrivacyServerSideFixedClipping(strategy,
                                                          configuration['differential_privacy']['noise_multiplier'],
                                                          configuration['differential_privacy']['clipping_norm'],
                                                          configuration['clients_per_round'])
    return strategy


def save_history_to_csv(history: History):
    data = {
        'Round': [item[0] for item in history.losses_centralized[1:]],
        'Loss (centralized)': [item[1].item() if not isinstance(item[1], float) else item[1] for item in history.losses_centralized[1:]],
    }
    for key, metric in history.metrics_centralized.items():
        data[f'{key} (centralized)'] = [item[1] for item in metric[1:]]
    for key, metric in history.metrics_distributed_fit.items():
        data[f'{key} (distributed, train)'] = [item[1] for item in metric]

    with open(os.path.join('logs', experiment_descriptor, 'history.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        writer.writeheader()
        writer.writerows([dict(zip(data.keys(), i)) for i in zip(*data.values())])


if __name__ == "__main__":
    if args.reproducible:
        fix_random(args.seed)

    if args.experiment_descriptor is None:
        experiment_descriptor = f"{configuration['attacker']['name'] if 'attacker' in configuration else 'benign'}-{configuration['defense']['name']}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    else:
        experiment_descriptor = args.experiment_descriptor

    if args.use_visdom:
        port = args.visdom_port if args.visdom_port is not None else 8097

    logger.log(INFO, f'{experiment_descriptor=}')

    if os.path.exists(os.path.join('logs', experiment_descriptor)):
        if args.y or confirm('This deletes the run\'s log data! Are you sure, you want to continue? [y / n] '):
            shutil.rmtree(os.path.join('logs', experiment_descriptor))
            if args.use_visdom:
                vis = Visdom(env=experiment_descriptor, port=port)
                vis.delete_env(experiment_descriptor)
        else:
            sys.exit(0)

    os.makedirs(os.path.join('logs', experiment_descriptor))

    # Configure logger
    logger.configure("Server", f"./logs/{experiment_descriptor}/log.log")
    logger.log(DEBUG, f"Training on {get_device()} using PyTorch {torch.__version__} and Flower {fl.__version__}")

    # Store configuration file alongside results and logs
    shutil.copyfile(args.params, f"./logs/{experiment_descriptor}/config.yaml")

    if args.use_visdom:
        # Start visdom environment for graphical result visualization
        vis = Visdom(env=experiment_descriptor, port=port)
        vis.text(yaml.dump(configuration).replace("\n", "<br>").replace("\t", "  "), win="Configuration")

    # Initialize csv file for intermediate result storage
    with open(os.path.join('logs', experiment_descriptor, 'history_intermediate.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Round', 'Accuracy', 'Backdoor Accuracy', 'Backdoored Accuracy', 'Misclassification Confidence', 'Malicious Clients'])
        writer.writeheader()

    # Initialize the inter-client communication helper
    client_communicator = ClientCommunicator.options(name="client_communicator", namespace="clients", lifetime="detached").remote()

    backdoor = get_backdoor(configuration) if 'train_benign_model' not in configuration else DummyBackdoor()

    criterion = get_criterion(configuration)

    if args.num_workers is not None:
        configuration['num_workers'] = args.num_workers
    num_workers = configuration.get('num_workers')

    task = ImageClassification(configuration['model'], configuration['dataset'], configuration['num_clients'],
                               configuration['batch_size'], criterion, experiment_descriptor,
                               configuration['partitioning'], configuration['partitioning_hyperparam'], configuration[
                                   'partitioning_balancing'] if 'partitioning_balancing' in configuration else False,
                               configuration['transformations'] if 'transformations' in configuration else None,
                               backdoor.backdoor_train_indices, backdoor.backdoor_test_indices,
                               configuration.get('dataset_fraction', 1.0), num_workers)

    # Rather ugly hack to allow the backdoor access to the transformations.
    # Nicer would be to pass task to backdoor but that's not possible since task needs the train and test indices from the backdoor.
    # Need to refactor this in the future to streamline all of this.
    backdoor.train_transformations = task.train_transformations
    backdoor.test_transformations = task.test_transformations

    if 'train_benign_model' in configuration:
        poisoning_scheduler = IntervalPoisoningScheduler(0, 0, 0)
    elif configuration['poisoning_scheduler']['name'] == 'frequency':
        poisoning_scheduler_args = [
            configuration['poisoning_scheduler']['start_round'],
            configuration['poisoning_scheduler']['end_round'],
            configuration['poisoning_scheduler']['frequency'],
            configuration['poisoning_scheduler']['malicious_clients_per_round'] if 'malicious_clients_per_round' in configuration['poisoning_scheduler'] else 1
        ]
        poisoning_scheduler = FrequencyPoisoningScheduler(*poisoning_scheduler_args)
    elif configuration['poisoning_scheduler']['name'] == 'interval':
        poisoning_scheduler = IntervalPoisoningScheduler(
            int(configuration['compromised_clients'] * configuration['num_clients']),
            configuration['poisoning_scheduler']['start_round'],
            configuration['poisoning_scheduler']['end_round'])
    elif configuration['poisoning_scheduler']['name'] == 'interval_fixed':
        poisoning_scheduler = IntervalFixedPoisoningScheduler(
            int(configuration['compromised_clients'] * configuration['num_clients']),
            configuration['poisoning_scheduler']['start_round'],
            configuration['poisoning_scheduler']['malicious_clients_per_round'] if 'malicious_clients_per_round' in configuration['poisoning_scheduler'] else int(configuration['compromised_clients'] * configuration['clients_per_round']),
            configuration['poisoning_scheduler']['end_round'])
    elif configuration['poisoning_scheduler']['name'] == 'round_wise':
        poisoning_scheduler = RoundWisePoisoningScheduler(configuration['poisoning_scheduler']['malicious_clients_dict'],)

    # Specify the resources each of your clients need. By default, each client will be allocated 1x CPU and 0x GPUs
    client_resources = {"num_cpus": 0.1, "num_gpus": 0.0}
    if get_device().type == "cuda":
        client_resources = {
            "num_cpus": configuration['resources']['cpus_per_client'],
            "num_gpus": configuration['resources']['gpus_per_client']
        }

    strategy = get_strategy()

    # Initialize a dummy client that can be used to evaluate the test set accuracy on the server side
    test_client = get_client("0")
    # Remove the client's logging handler, as this would just clutter the logs (this is kind of hacky, but okay as a one-time solution)
    logger.logger.removeHandler(test_client.logging_handler)

    if args.test_batch_size:
        configuration['test_batch_size'] = args.test_batch_size

    test_batch_size = configuration.get('test_batch_size', 1024)
    test_loader = task.get_dataloader(None, "test", test_batch_size)

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=configuration['num_clients'],
        config=fl.server.ServerConfig(
            num_rounds=configuration['num_rounds'],
        ),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args={
            'namespace': 'clients',
            "ignore_reinit_error": True,
        },
        keep_initialised=True,
        client_manager=CustomClientManager(poisoning_scheduler),
    )

    save_history_to_csv(history)

    lifespan_thresholds, lifespans, mean_lifespan, weighted_mean_lifespan = compute_lifespans(history, poisoning_scheduler.get_last_poisoning_round(configuration['num_rounds']), configuration)
    mean_lifespan_artificial_trigger, weighted_mean_lifespan_artificial_trigger = np.mean(lifespans[2:]), np.average(lifespans[2:], weights=np.arange(0.15, 1.01, 0.05))
    auc_mta_during, auc_bda_during, auc_mta_after, auc_bda_after, auc_mta_during_after, auc_bda_during_after = compute_auc(history, poisoning_scheduler.get_first_poisoning_round(), poisoning_scheduler.get_last_poisoning_round(configuration['num_rounds']), configuration)
    auc_mta_artificial_trigger_during, auc_bda_artificial_trigger_during, auc_mta_artificial_trigger_after, auc_bda_artificial_trigger_after, auc_mta_artificial_trigger_during_after, auc_bda_artificial_trigger_during_after = compute_auc(history, poisoning_scheduler.get_first_poisoning_round(), poisoning_scheduler.get_last_poisoning_round(configuration['num_rounds']), configuration, 0.1)
    with open(f"./logs/{experiment_descriptor}/lifespans.log", "w") as f:
        for threshold, lifespan in zip(lifespan_thresholds, lifespans):
            f.writelines(f'{threshold * 100:3.0f} %:\t{lifespan}\n')
        f.write(f"\nMean Lifespan [5%, 100%]: {mean_lifespan}")
        f.write(f"\nWeighted Mean Lifespan [5%, 100%]: {weighted_mean_lifespan}")
        f.write(f"\nMean Lifespan [15%, 100%]: {mean_lifespan_artificial_trigger}")
        f.write(f"\nWeighted Mean Lifespan [15%, 100%]: {weighted_mean_lifespan_artificial_trigger}")
        f.write(f"\nAuC during attack window [0%, 100%] (MTA / BDA): {auc_mta_during} / {auc_bda_during}")
        f.write(f"\nAuC after attack window [0%, 100%] (MTA / BDA): {auc_mta_after} / {auc_bda_after}")
        f.write(f"\nAuC during & after attack window [0%, 100%] (MTA / BDA): {auc_mta_during_after} / {auc_bda_during_after}")
        f.write(f"\nAuC during the attack window [10%, 100%] (MTA / BDA): {auc_mta_artificial_trigger_during} / {auc_bda_artificial_trigger_during}")
        f.write(f"\nAuC after attack window [10%, 100%] (MTA / BDA): {auc_mta_artificial_trigger_after} / {auc_bda_artificial_trigger_after}")
        f.write(f"\nAuC during & after attack window [10%, 100%] (MTA / BDA): {auc_mta_artificial_trigger_during_after} / {auc_bda_artificial_trigger_during_after}")

    avg_accs_before, avg_accs_during, avg_accs_after = compute_average_mtas(history,
                                                                            configuration['poisoning_scheduler']['start_round'],
                                                                            configuration['poisoning_scheduler']['end_round'],
                                                                            configuration)

    with open(f"./logs/{experiment_descriptor}/average_accs.log", "w") as f:
        f.write(f'Accs before attack window (MTA / BDA / BACC): {avg_accs_before[0]} / {avg_accs_before[1]} / {avg_accs_before[2]}\n')
        f.write(f'Accs during attack window (MTA / BDA / BACC): {avg_accs_during[0]} / {avg_accs_during[1]} / {avg_accs_during[2]}\n')
        f.write(f'Accs after attack window (MTA / BDA / BACC): {avg_accs_after[0]} / {avg_accs_after[1]} / {avg_accs_after[2]}')

    training_accuracies = np.array([0] + [acc[1] for acc in history.metrics_distributed_fit['accuracy']])

    if args.use_visdom:
        vis.line(training_accuracies, [round if 'load_benign_model' not in configuration else configuration['load_benign_model']['trained_rounds'] + round for round in range(len(training_accuracies))], win="acc", update="append", name="train acc.")

        vis.save([experiment_descriptor])
        shutil.copyfile(os.path.join(os.path.expanduser('~'), '.visdom', f'{experiment_descriptor}.json'), os.path.join('logs', experiment_descriptor, 'visdom.json'))
