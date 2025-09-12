import argparse
import json
import logging
import os
import subprocess
from typing import Dict, List, Tuple

from optuna.samplers import NSGAIISampler
import optuna
from optuna.trial import FrozenTrial

from evaluation.hyper_params.utils import extract_accuracies
from utils import fix_random


def run_batch(configurations: List[Dict]):
    cmd = ["python",
           "multi_run.py",
           "--base_config", args.base_config,
           "--lr_b&wd_b&E_b&B_b&beta&wd_m&E_m&B_m"]

    for config in configurations:
        if not config['already_tested']:
            cmd += [str(config['lr_b']), str(config['wd_b']), str(config['E_b']), str(config['B_b']), str(config['beta']), str(config['wd_m']), str(config['E_m']), str(config['B_m']),]

    cmd += [ "--experiment_descriptor_template", f"{args.experiment_descriptor_prefix}_{{lr_b}}_{{wd_b}}_{{E_b}}_{{B_b}}/{{beta}}_{{wd_m}}_{{E_m}}_{{B_m}}",
              "--devices"] + [str(device) for device in args.devices] + [
              "--tasks_per_gpu"] + [str(task_num) for task_num in args.tasks_per_gpu] + [
              "--test_batch_size", str(args.test_batch_size),
              "--num_workers", str(args.num_workers),]

    print(cmd)

    try:
        subprocess.check_output(cmd, env=dict(os.environ), stderr=subprocess.STDOUT, cwd=os.getcwd())
    except subprocess.CalledProcessError as e:
        print(e.output)
        logging.error(e.output)

    mtas = []
    bdas = []
    for config in configurations:
        path = f"logs/{args.experiment_descriptor_prefix}_{config['lr_b']}_{config['wd_b']}_{config['E_b']}_{config['B_b']}/{config['beta']}_{config['wd_m']}_{config['E_m']}_{config['B_m']}"
        if os.path.exists(f'{path}/average_accs.log'):
            results = extract_accuracies(path)
            mtas.append(results["during_attack_window"]["MTA"])
            bdas.append(results["during_attack_window"]["BDA"])
        else:
            # The worker crashed or something else went wrong
            mtas.append(0.0)
            bdas.append(0.0)

    return [(mta, bda) for mta, bda in zip(mtas, bdas)]

def mta_constraint(trial: FrozenTrial) -> List[float]:
    return [mta_threshold - trial.user_attrs["MTA"]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Approximate the Pareto front using NSGA-II")
    parser.add_argument("--experiment_descriptor_prefix", type=str, help="Prefix for all experiment descriptors. Remaining part will be auto-generated.")
    parser.add_argument("--base_config", required=True, help="Base config file")
    parser.add_argument("--gpus_per_client", default=".", type=str, help="Number of GPUs required per FL client")
    parser.add_argument("--cpus_per_client", default=".", type=str, help="Number of CPUs required per FL client")
    parser.add_argument('--devices', type=int, nargs='+', required=True, help="Space-separated list of GPU indices to use")
    parser.add_argument('--tasks_per_gpu', type=int, nargs='+', required=True, help="Space-separated list of tasks per GPU (must have same length as --devices)")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of dataloader workers")
    parser.add_argument('--test_batch_size', type=int, help="The test batch size to use", default=1024)
    parser.add_argument('--population_size_malicious', type=int, help="The number of individuals to create in each generation of the malicious NSGA-II optimization run", default=4)
    parser.add_argument('--generations_malicious', type=int, help="The number of generations to run the malicious NSGA-II optimization for", default=3)
    parser.add_argument('--ideal_mta', type=float, help="The maximum MTA an undefended and unattacked model achieves", default=0.8606)
    parser.add_argument('--mta_diff', type=float, help="The maximum allowed decrease in MTA caused by the attack. Do not set to enable unconstrained optimization.")
    parser.add_argument('--reproducible', action="store_true", help="Fix all random seeds")
    parser.add_argument('--lr_b', type=float, help="The benign learning rate", default=0.15)
    parser.add_argument('--wd_b', type=float, help="The benign weight decay", default=0.0005)
    parser.add_argument('--E_b', type=int, help="The benign local epochs", default=10)
    parser.add_argument('--B_b', type=int, help="The benign batch size", default=32)

    args = parser.parse_args()

    if args.reproducible:
        fix_random()

    mta_constrained = args.mta_diff is not None and args.mta_diff > 0
    if mta_constrained:
        mta_threshold = args.ideal_mta - args.mta_diff

    os.makedirs(f"logs/{args.experiment_descriptor_prefix}", exist_ok=True)

    generations_to_configurations = {}

    logging.basicConfig(
        handlers=[
            logging.FileHandler(f"logs/{args.experiment_descriptor_prefix}/NSGA-II.log"),
            logging.StreamHandler()
        ],
        level=logging.DEBUG
    )

    # Start one NSGA-II instance per benign configuration to tweak the malicious parameters
    mal_study = optuna.create_study(
        directions=["maximize", "maximize"],  # Adversary wants to maximize BDA and MTA
        sampler=NSGAIISampler(constraints_func=mta_constraint, population_size=args.population_size_malicious),
    ) if mta_constrained else optuna.create_study(
        directions=["maximize", "maximize"],  # Adversary wants to maximize BDA and MTA
        sampler=NSGAIISampler(population_size=args.population_size_malicious),
    )

    for gen_mal in range(args.generations_malicious):
        logging.info(f'Starting malicious generation {gen_mal} / {args.generations_malicious}')
        configurations = []
        mal_trials = []

        for _ in range(args.population_size_malicious):
            trial = mal_study.ask()
            config = {
                "beta": trial.suggest_float(name="beta", low=0.2, high=10),
                "wd_m": trial.suggest_float(name="wd_m", low=0.0001, high=0.001),
                "E_m": trial.suggest_int(name="E_m", low=2, high=20),
                "B_m": trial.suggest_int(name="B_m", low=16, high=128, step=16),
            }
            config.update({
                "lr_b": args.lr_b,
                "wd_b": args.wd_b,
                "E_b": args.E_b,
                "B_b": args.B_b,
            })
            experiment_str = f"{args.experiment_descriptor_prefix}_{config['lr_b']}_{config['wd_b']}_{config['E_b']}_{config['B_b']}/{config['beta']}_{config['wd_m']}_{config['E_m']}_{config['B_m']}"
            already_tested = False
            for k, v in generations_to_configurations.items():
                if experiment_str in v:
                    already_tested = True
                    logging.info(f'Config {config} in {gen_mal=} was already tested in generation {k}.')
            config['already_tested'] = already_tested

            mal_trials.append(trial)
            configurations.append(config)

        logging.info(f'Malicious configurations in malicious generation {gen_mal} are {configurations}')

        generations_to_configurations[gen_mal] = [f"{args.experiment_descriptor_prefix}_{config['lr_b']}_{config['wd_b']}_{config['E_b']}_{config['B_b']}/{config['beta']}_{config['wd_m']}_{config['E_m']}_{config['B_m']}" for config in configurations]

        with open(f"logs/{args.experiment_descriptor_prefix}/generations_to_configurations.json", "w") as f:
            json.dump(generations_to_configurations, f)

        objectives = run_batch(configurations)

        # finish all malicious trials
        for j, trial in enumerate(mal_trials):
            mta, bda = objectives[j]
            logging.debug(f'In {gen_mal=} {trial.number=} yielded MTA: {mta} BDA: {bda}')
            trial.set_user_attr("MTA", mta)
            mal_study.tell(trial, (mta, bda))

        pareto_trials = mal_study.best_trials
        if len(pareto_trials) == 0: # No trial fulfilled the constraint or something crashed
            logging.warning(f'There was no valid malicious solution in {gen_mal=} resorting to reporting trial with highest MTA')
            pareto_trials = [max(mal_study.trials, key=lambda t: t.values[0])]
        logging.debug(f'Best adversary in {gen_mal=} achieved (MTA/BDA) = {max(pareto_trials, key=lambda x: x.values[1]).values}')

    # Access the Pareto front
    pareto_trials = mal_study.best_trials
    for t in pareto_trials:
        logging.info(f"Params: {t.params}, MTA: {t.values[0]}, BDA: {t.values[1]}")
