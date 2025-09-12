import argparse
import itertools
import json
import os
import subprocess
import tempfile
import threading
import time

import math
import numpy as np
import yaml


class Worker(threading.Thread):
    def __init__(self, sid, gpu, task_provider, visdom_port, use_visdom, test_batch_size, num_workers):
        threading.Thread.__init__(self)
        self.gpu = gpu
        self.tp = task_provider
        self.id = sid
        self.task_nr = 0
        self.visdom_port = visdom_port
        self.use_visdom = use_visdom
        self.num_workers = num_workers
        self.test_batch_size = test_batch_size

    def run(self):
        while self.tp.more_tasks():
            self.task_nr += 1
            configuration, experiment_descriptor, seed = self.tp.get_task()

            if '{counter}' in experiment_descriptor:
                experiment_descriptor += f'_{self.id}_{self.task_nr}_{int(time.time())}'

            tf = tempfile.NamedTemporaryFile()
            with open(tf.name, "w") as f:
                yaml.dump(configuration, f, default_flow_style=False)

                t = time.time()
                print(f"Runner-{self.id}: {tf.name}")

                cmd = ['python', 'FLSimulation.py',
                       '--params', tf.name,
                       '--devices', str(self.gpu),
                       '-y',
                       '--reproducible',
                       '--experiment_descriptor', experiment_descriptor,
                       '--seed', str(seed),
                       '--num_workers', str(self.num_workers),]

                if self.visdom_port:
                    cmd.extend(['--visdom_port', str(self.visdom_port)])

                if self.test_batch_size:
                    cmd.extend(['--test_batch_size', str(self.test_batch_size)])

                if self.use_visdom:
                    cmd.extend(['--use_visdom'])

                print(cmd)

                try:
                    subprocess.check_output(cmd, env=dict(os.environ), stderr=subprocess.STDOUT, cwd=os.getcwd())
                except subprocess.CalledProcessError as e:
                    print(e.output)

                t2 = time.time()
                print(f"Runner-{self.id} Execution time: {int((t2 - t) / 36) / 100} hours")


class TaskManager:
    def __init__(self, tasks):
        self.lock = threading.Lock()

        print("No. of tasks: {}".format(len(tasks)))
        self.tasks = tasks
        self.idx = 0

    def more_tasks(self):
        return self.idx < len(self.tasks)

    def get_task(self):
        # lock
        with self.lock:
            if self.idx >= len(self.tasks):
                return "echo Tasks finished for thread on GPU {}"
            task = self.tasks[self.idx]
            self.idx += 1
            print(f"Dispatched task {self.idx} / {len(self.tasks)}")
            return task


def parse_mal(mal, ben_val, lower_bound, upper_bound):
    if isinstance(mal, str) and '*' in mal:
        mal = min(max(float(mal.replace('*', '')) * ben_val, lower_bound), upper_bound)

    if isinstance(ben_val, float):
        mal = float(mal)
    elif isinstance(ben_val, int):
        mal = int(mal)

    if mal > upper_bound:
        mal = upper_bound
    if mal < lower_bound:
        mal = lower_bound

    return mal


def interpolate_value(data, attack, ben_val):
    if attack not in data:
        raise ValueError(f"Attack {attack} not found in data.")

    ben_values = list(data[attack].keys())
    is_float = '.' in ben_values[0]
    if not is_float:
        ben_values = np.array(sorted([int(v) for v in ben_values]))
    else:
        ben_values = np.array(sorted([float(v) for v in ben_values]))

    closest_val = ben_values[np.argmin(np.abs(ben_values - ben_val))]

    return data[attack][str(closest_val)]


def infer_type(val):
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val


def parse_unknown_args(unknown_args):
    parsed = {}
    key = None
    for arg in unknown_args:
        if arg.startswith('--'):
            key = arg[2:]
            parsed[key] = []  # Start collecting values
        elif key:
            parsed[key].append(arg)
        else:
            raise ValueError(f"Unexpected value without flag: {arg}")

    # Process zipped parameters
    zipped = {}
    zipped_keys = []
    for key, vals in parsed.items():
        if '&' in key:
            keys = key.split('&')
            zipped_keys.append(key)
            grouped = [infer_type(v) for v in vals]
            tuples = list(zip(*[grouped[i::len(keys)] for i in range(len(keys))]))
            for i, k in enumerate(keys):
                zipped[k] = [t[i] for t in tuples]
        else:
            zipped[key] = [infer_type(v) for v in vals]

    return zipped, zipped_keys


def expand_args_grid(args_dict, zipped_keys=None):
    zipped_keys = zipped_keys or []
    zipped_groups = [k.split('&') for k in zipped_keys]

    # Split zipped and non-zipped keys
    used_keys = set(k for group in zipped_groups for k in group)
    non_zipped = {
        k: v if isinstance(v, list) else [v]
        for k, v in args_dict.items()
        if k not in used_keys
    }

    zipped_products = []
    for group in zipped_groups:
        zipped_values = list(zip(*(args_dict[k] for k in group)))
        zipped_products.append((group, zipped_values))

    # Build grid
    for non_zip_combo in itertools.product(*non_zipped.values()):
        combined = dict(zip(non_zipped.keys(), non_zip_combo))
        if len(zipped_products) == 0:
            yield combined
        else:
            for (zipped_group, zipped_vals) in zipped_products:
                for vals in zipped_vals:
                    for (k, v) in zip(zipped_group, vals):
                        combined[k] = v
                    yield combined


def set_nested(config, key_path, value, sep='/'):
    keys = key_path.split(sep)
    d = config
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def flatten_dict(d, parent_key='', sep='/'):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLSimulation Multi Run")
    parser.add_argument("--base_config", required=True, help="Path to base config file")
    parser.add_argument("--gpus_per_client", default=".", type=str, help="Number of GPUs required per FL client")
    parser.add_argument("--cpus_per_client", default=".", type=str, help="Number of CPUs required per FL client")
    parser.add_argument('--experiment_descriptor_prefix', type=str, required=False, help="Prefix to add to all experiment descriptors - rest of the experiment descriptor will simply count upwards")
    parser.add_argument('--experiment_descriptor_template', type=str, required=False, help="Template for experiment descriptors. You can use every variable from the configuration file as a placeholder.")
    parser.add_argument('--devices', type=int, nargs='+', required=True, help="Space-separated list of GPU indices to use")
    parser.add_argument('--tasks_per_gpu', type=int, nargs='+', required=True, help="Space-separated list of tasks per GPU (must have same length as --devices)")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of dataloader workers")
    parser.add_argument('--visdom_port', type=int, default=8097, help="Port on which the visdom server runs")
    parser.add_argument('--use_visdom', action="store_true", help="Visualize results using visdom.")
    parser.add_argument('--test_batch_size', type=int, help="The test batch size to use.", default=1024)
    parser.add_argument('--malicious_strategy', type=str, choices=['greedy', 'balanced', 'MTA_constrained_0.02', 'MTA_constrained_0.05'],  help="The strategy that is used to determine the malicious hyperparameters", default='MTA_constrained')
    parser.add_argument("--seed", default=42, type=int, help="A seed for all random number generators.", nargs='+')

    args, unknown = parser.parse_known_args()
    parsed_unknown_args, zipped_keys = parse_unknown_args(unknown)

    tasks = []

    attack_lower_case_to_name = {
        'a3fl': 'A3FL',
        'chameleon': 'Chameleon',
        'data_poisoning': 'DataPoisoning',
        'iba': 'IBA',
        'fcba': 'FCBA',
        'darkfed': 'DarkFed',
    }

    mapping = { # For backwards compatibility
        'lr_b': 'optimizer/local_lr',
        'beta': 'adversarial_optimizer/lr_factor',
        'mu_b': 'optimizer/momentum',
        'mu_m': 'adversarial_optimizer/momentum',
        'wd_b': 'optimizer/weight_decay',
        'wd_m': 'adversarial_optimizer/weight_decay',
        'E_b': 'local_epochs',
        'E_m': 'local_epochs_malicious_clients',
        'B_b': 'batch_size',
        'B_m': 'attacker/batch_size',
        'alpha': 'compromised_clients',
        'N': 'num_clients',
        'M': 'clients_per_round',
        'pre_trained_model': 'load_benign_model/experiment_descriptor',
    }

    seeds = args.seed if isinstance(args.seed, list) else [args.seed]

    for seed in seeds:
        for combo in expand_args_grid(parsed_unknown_args, zipped_keys):

            # Load base config file
            with open(args.base_config, "r") as file:
                configuration = yaml.safe_load(file)

            attack = attack_lower_case_to_name[configuration['attacker']['name']]

            for key, val in combo.items():
                if key == 'dirichlet':
                    if val == 'iid':
                        set_nested(configuration, 'partitioning', 'iid')
                    else:
                        set_nested(configuration, 'partitioning', 'dirichlet')
                        set_nested(configuration, 'partitioning_hyperparam', float(val))
                    continue

                if key == 'beta':
                    if val == '?':
                        with open(f'evaluation/hyper_params/results/best_mal_choices_{args.malicious_strategy}_lr.json', 'r') as f:
                            best_vals = json.load(f)
                            val = interpolate_value(best_vals, attack, configuration['optimizer']['local_lr'])
                    else:
                        val = float(val)

                    configuration.get('adversarial_optimizer', {}).pop('local_lr', None)
                    configuration.get('adversarial_optimizer', {}).pop('lr_factor', None)

                    key = 'adversarial_optimizer/lr_factor'

                    # Account for the Chameleon attack using two optimizers
                    if configuration.get('attacker').get('name').lower() == 'chameleon':
                        configuration.get('attacker').get('contrastive_optimizer', {}).pop('local_lr', None)
                        configuration.get('attacker').get('contrastive_optimizer', {}).pop('lr_factor', None)

                        set_nested(configuration, 'attacker/contrastive_optimizer/lr_factor', val)
                elif key == 'mu_m':
                    if val == '?':
                        with open(f'evaluation/hyper_params/results/best_mal_choices_{args.malicious_strategy}_mu.json', 'r') as f:
                            best_vals = json.load(f)
                            val = interpolate_value(best_vals, attack, configuration['optimizer']['momentum'])
                    else:
                        val = parse_mal(val, configuration['optimizer']['momentum'], 0, 1)

                    # Account for the Chameleon attack using two optimizers
                    if configuration.get('attacker').get('name').lower() == 'chameleon':
                        set_nested(configuration, 'attacker/contrastive_optimizer/momentum', val)
                elif key == 'wd_m':
                    if val == '?':
                        with open(f'evaluation/hyper_params/results/best_mal_choices_{args.malicious_strategy}_wd.json', 'r') as f:
                            best_vals = json.load(f)
                            val = interpolate_value(best_vals, attack, configuration['optimizer']['weight_decay'])
                    else:
                        val = parse_mal(val, configuration['optimizer']['weight_decay'], 0, 1)

                    # Account for the Chameleon attack using two optimizers
                    if configuration.get('attacker').get('name').lower() == 'chameleon':
                        set_nested(configuration, 'attacker/contrastive_optimizer/weight_decay', val)
                elif key == 'E_m':
                    if val == '?':
                        with open(f'evaluation/hyper_params/results/best_mal_choices_{args.malicious_strategy}_E.json', 'r') as f:
                            best_vals = json.load(f)
                            val = interpolate_value(best_vals, attack, configuration['local_epochs'])
                    else:
                        val = parse_mal(val, configuration['local_epochs'], 1, math.inf)

                    # Account for the adversarial learning rate scheduler of DarkFed
                    if configuration.get('attacker').get('name').lower() == 'darkfed':
                        set_nested(configuration, 'adversarial_lr_scheduler/T_max', val)
                elif key == 'B_m':
                    if val == '?':
                        with open(f'evaluation/hyper_params/results/best_mal_choices_{args.malicious_strategy}_B.json', 'r') as f:
                            best_vals = json.load(f)
                            val = interpolate_value(best_vals, attack, configuration['batch_size'])
                    else:
                        val = parse_mal(val, configuration['batch_size'], 1, 512)

                set_nested(configuration, mapping[key] if key in mapping else key, val)

            if args.gpus_per_client != '.':
                configuration['resources']['gpus_per_client'] = float(args.gpus_per_client)
            if args.cpus_per_client != '.':
                configuration['resources']['cpus_per_client'] = float(args.cpus_per_client)

            flattened_configuration = flatten_dict(configuration)
            template_keys_values = {
                f'{{{k}}}': str(v) for k, v in flattened_configuration.items()
            }
            template_keys_values.update({
                f'{{{alternative_key}}}': str(flattened_configuration[mapping[alternative_key]]) for alternative_key in mapping.keys() if mapping[alternative_key] in flattened_configuration.keys()
            })
            template_keys_values.update({'{seed}': seed})

            if args.experiment_descriptor_template:
                experiment_descriptor = args.experiment_descriptor_template
                for (key, value) in template_keys_values.items():
                    experiment_descriptor = experiment_descriptor.replace(key, str(value))
            else:
                experiment_descriptor = args.experiment_descriptor_prefix + '{counter}'

            if 'load_benign_model' in configuration:
                for (key, value) in template_keys_values.items():
                    configuration['load_benign_model']['experiment_descriptor'] = configuration['load_benign_model']['experiment_descriptor'].replace(key, str(value))

            tasks.append((configuration, experiment_descriptor, seed))

    tm = TaskManager(tasks)

    workers = []

    counter = 0
    for gpu, num_tasks in zip(args.devices, args.tasks_per_gpu):
        for i in range(num_tasks):
            workers.append(Worker(counter, gpu, tm, args.visdom_port, args.use_visdom, args.test_batch_size, args.num_workers))
            counter += 1

    for w in workers:
        w.start()
    for w in workers:
        w.join()
