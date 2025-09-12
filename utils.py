import csv
import os
import inspect
import random
from collections import defaultdict

import numpy as np
import torch

from flwr.server import History
from sklearn.metrics import auc


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def confirm(msg):
    answer = input(msg)
    while answer.lower() not in ['y', 'n', 'yes', 'no']:
        answer = input("Please enter 'y' or 'n': ")
    return answer.lower() in ['y', 'yes']


def fix_random(seed : int = 42):
    from torch.backends import cudnn
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.enabled = True
    cudnn.benchmark = False
    np.random.seed(seed)

def is_method_implemented(cls, method_name):
    method = getattr(cls, method_name, None)
    if method is None:
        return False  # Method does not exist

    try:
        source = inspect.getsource(method).strip()
        return "pass" not in source  # Returns False if 'pass' is the only content
    except TypeError:
        return False  # In case the method is not a regular function (e.g., built-in)

def sliding_average(vals, omega):
    return [np.mean(vals[max(0, i - omega):min(len(vals), i + omega + 1)]) for i in range(len(vals))]

def get_rounds(history: History, configuration):
    offset = 0
    if 'load_benign_model' in configuration:
        first_round, _ = history.metrics_centralized['backdoor_accuracy'][0]
        if first_round < configuration['load_benign_model']['trained_rounds']:
            offset = configuration['load_benign_model']['trained_rounds'] - first_round

    return [round + offset for round, _ in history.metrics_centralized['backdoor_accuracy']]

def compute_lifespans(history: History, last_poisoning_round: int, configuration, thresholds = np.arange(0.05, 1.01, 0.05), omega: int = 10):
    first_poisoning_rounds = (last_poisoning_round) * np.ones_like(thresholds)
    last_poisoning_rounds = (last_poisoning_round) * np.ones_like(thresholds)

    rounds = get_rounds(history, configuration)

    backdoor_accuracies = [backdoor_accuracy for _, backdoor_accuracy in history.metrics_centralized['backdoor_accuracy']]

    index_of_first_non_poisoning_round = rounds.index(last_poisoning_round) + 1

    # Compute a sliding average over the backdoor accuracies to smoothen out any peaks
    backdoor_accuracies[index_of_first_non_poisoning_round:] = sliding_average(backdoor_accuracies[index_of_first_non_poisoning_round:], omega)

    for round, backdoor_accuracy in zip(rounds, backdoor_accuracies):
        if round > last_poisoning_round:
            # Determine the first and last successful round for each backdoor accuracy threshold level
            first_poisoning_rounds = np.where((first_poisoning_rounds == last_poisoning_round) * (last_poisoning_rounds == last_poisoning_round) * (backdoor_accuracy >= thresholds), round, first_poisoning_rounds)
            last_poisoning_rounds = np.where(backdoor_accuracy >= thresholds, round, last_poisoning_rounds)

    lifespans = np.where(last_poisoning_rounds > last_poisoning_round, last_poisoning_rounds + 1 - first_poisoning_rounds, 0)

    return thresholds, lifespans, np.mean(lifespans), np.average(lifespans, weights=thresholds)


def compute_auc(history: History, first_poisoning_round, last_poisoning_round: int, configuration, baseline: float = 0.0):
    rounds = get_rounds(history, configuration)

    backdoor_accuracies = [backdoor_accuracy for _, backdoor_accuracy in history.metrics_centralized['backdoor_accuracy']]
    clean_accuracies = [clean_accuracy for _, clean_accuracy in history.metrics_centralized['accuracy']]

    index_of_first_non_poisoning_round = rounds.index(last_poisoning_round) + 1
    index_of_first_poisoning_round = rounds.index(first_poisoning_round)

    # Compute a sliding average over the backdoor accuracies to smoothen out any peaks
    backdoor_accuracies_after = backdoor_accuracies[index_of_first_non_poisoning_round:]
    clean_accuracies_after = clean_accuracies[index_of_first_non_poisoning_round:]
    backdoor_accuracies_during = backdoor_accuracies[index_of_first_poisoning_round:index_of_first_non_poisoning_round]
    clean_accuracies_during = clean_accuracies[index_of_first_poisoning_round:index_of_first_non_poisoning_round]
    backdoor_accuracies_during_after = backdoor_accuracies[index_of_first_poisoning_round:]
    clean_accuracies_during_after = clean_accuracies[index_of_first_poisoning_round:]

    return (auc(rounds[index_of_first_poisoning_round:index_of_first_non_poisoning_round], (np.array(clean_accuracies_during) - baseline)),
            auc(rounds[index_of_first_poisoning_round:index_of_first_non_poisoning_round], (np.array(backdoor_accuracies_during) - baseline)),
            auc(rounds[index_of_first_non_poisoning_round:], (np.array(clean_accuracies_after) - baseline)),
            auc(rounds[index_of_first_non_poisoning_round:], (np.array(backdoor_accuracies_after) - baseline)),
            auc(rounds[index_of_first_poisoning_round:], (np.array(clean_accuracies_during_after) - baseline)),
            auc(rounds[index_of_first_poisoning_round:], (np.array(backdoor_accuracies_during_after) - baseline)))


def compute_average_mtas(history: History, start_round: int, end_round: int, configuration):
    rounds = [round if 'load_benign_model' not in configuration else round + configuration['load_benign_model']['trained_rounds'] for round, _ in history.metrics_centralized['accuracy']]
    accuracies = [accuracy for _, accuracy in history.metrics_centralized['accuracy']]
    backdoor_accuracies = [backdoor_accuracy for _, backdoor_accuracy in history.metrics_centralized['backdoor_accuracy']]
    backdoored_accuracies = [backdoored_accuracy for _, backdoored_accuracy in history.metrics_centralized['backdoored_accuracy']]

    attack_start_index, attack_end_index = rounds.index(start_round), rounds.index(end_round)

    avg_mta_before, avg_bda_before, avg_backdoored_acc_before = np.mean(accuracies[:attack_start_index]), np.mean(backdoor_accuracies[:attack_start_index]), np.mean(backdoored_accuracies[:attack_start_index])
    avg_mta_during, avg_bda_during, avg_backdoored_acc_during = np.mean(accuracies[attack_start_index:attack_end_index + 1]), np.mean(backdoor_accuracies[attack_start_index:attack_end_index + 1]), np.mean(backdoored_accuracies[attack_start_index:attack_end_index + 1])
    avg_mta_after, avg_bda_after, avg_backdoored_acc_after = np.mean(accuracies[attack_end_index + 1:]), np.mean(backdoor_accuracies[attack_end_index + 1:]), np.mean(backdoored_accuracies[attack_end_index + 1:])

    return (avg_mta_before, avg_bda_before, avg_backdoored_acc_before), (avg_mta_during, avg_bda_during, avg_backdoored_acc_during), (avg_mta_after, avg_bda_after, avg_backdoored_acc_after)


def restore_history_from_csv(experiment_descriptor: str) -> History:
    history = History()

    with open(os.path.join('logs', experiment_descriptor, 'history.csv'), 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Initialize dictionaries to hold the metrics
    losses_centralized = []
    metrics_centralized = {}
    metrics_distributed_fit = {}

    for row in rows:
        round_num = int(row['Round'])
        loss_centralized = float(row['Loss (centralized)'])
        losses_centralized.append((round_num, loss_centralized))

        for key in row:
            if key not in ['Round', 'Loss (centralized)']:
                value = float(row[key])
                if '(centralized)' in key:
                    metric_key = key.replace(' (centralized)', '')
                    if metric_key not in metrics_centralized:
                        metrics_centralized[metric_key] = []
                    metrics_centralized[metric_key].append((round_num, value))
                elif '(distributed, train)' in key:
                    metric_key = key.replace(' (distributed, train)', '')
                    if metric_key not in metrics_distributed_fit:
                        metrics_distributed_fit[metric_key] = []
                    metrics_distributed_fit[metric_key].append((round_num, value))

    # Adjust the losses_centralized to match the expected structure (with dummy first entry)
    if losses_centralized:
        history.losses_centralized = [(0, 0.0)] + losses_centralized
    else:
        history.losses_centralized = []

    # Adjust metrics_centralized to match the expected structure (with dummy first entry)
    for key in metrics_centralized:
        history.metrics_centralized[key] = [(0, 0.0)] + metrics_centralized[key]

    history.metrics_distributed_fit = metrics_distributed_fit

    return history


def restore_history_from_intermediate_csv(experiment_descriptor: str) -> History:
    history = History()

    with open(os.path.join('logs', experiment_descriptor, 'history_intermediate.csv'), 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Initialize dictionaries to hold the metrics
    losses_centralized = []
    metrics_centralized = defaultdict(list)
    metrics_distributed_fit = defaultdict(list)

    for row in rows:
        round_num = int(row['Round'])
        metrics_centralized['accuracy'].append((round_num, float(row['Accuracy'])))
        metrics_centralized['backdoor_accuracy'].append((round_num, float(row['Backdoor Accuracy'])))
        metrics_centralized['backdoored_accuracy'].append((round_num, float(row['Backdoored Accuracy'])))
        metrics_centralized['malicious_clients'].append((round_num, int(row['Malicious Clients'])))

    # Adjust the losses_centralized to match the expected structure (with dummy first entry)
    if losses_centralized:
        history.losses_centralized = [(0, 0.0)] + losses_centralized
    else:
        history.losses_centralized = []

    # Adjust metrics_centralized to match the expected structure (with dummy first entry)
    for key in metrics_centralized:
        history.metrics_centralized[key] = metrics_centralized[key]

    history.metrics_distributed_fit = metrics_distributed_fit

    return history
