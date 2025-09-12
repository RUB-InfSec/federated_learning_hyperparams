import argparse
import json
import logging
import os
import subprocess
from typing import Dict, List

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
    parser.add_argument('--population_size_benign', type=int, help="The number of individuals to create in each generation of the benign NSGA-II optimization run", default=4)
    parser.add_argument('--population_size_malicious', type=int, help="The number of individuals to create in each generation of the malicious NSGA-II optimization run", default=4)
    parser.add_argument('--generations_benign', type=int, help="The number of generations to run the benign NSGA-II optimization for", default=3)
    parser.add_argument('--generations_malicious', type=int, help="The number of generations to run the malicious NSGA-II optimization for", default=3)
    parser.add_argument('--ideal_mta', type=float, help="The maximum MTA an undefended and unattacked model achieves", default=0.8606)
    parser.add_argument('--mta_diff', type=float, help="The maximum allowed decrease in MTA caused by an attack or defense. Do not set to enable unconstrained optimization.")
    parser.add_argument('--reproducible', action="store_true", help="Fix all random seeds")

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

    if mta_constrained:
        ben_study = optuna.create_study(
            directions=["maximize", "minimize"], # Maximize MTA while minimizing BDA
            sampler=NSGAIISampler(constraints_func=mta_constraint, population_size=args.population_size_benign),
        )
    else:
        ben_study = optuna.create_study(
            directions=["maximize", "minimize"], # Maximize MTA while minimizing BDA
            sampler=NSGAIISampler(population_size=args.population_size_benign),
        )

    for gen_ben in range(args.generations_benign):
        logging.info(f'Starting benign generation {gen_ben} / {args.generations_benign}')
        # create batch
        ben_trials = []
        benign_configurations = []

        for _ in range(args.population_size_benign):
            trial = ben_study.ask()
            ben_trials.append(trial)
            config = {
                "lr_b": trial.suggest_float(name="lr_b", low=0.05, high=0.5),
                "wd_b": trial.suggest_float(name="wd_b", low=0.0001, high=0.001),
                "E_b": trial.suggest_int(name="E_b", low=2, high=20),
                "B_b": trial.suggest_int(name="B_b", low=16, high=128, step=16),
            }
            benign_configurations.append(config)

        ben_objectives = []

        logging.info(f'Benign configurations in benign generation {gen_ben} are {benign_configurations}')
        generations_to_configurations[gen_ben] = {}

        # Start one NSGA-II instance per benign configuration to tweak the malicious parameters
        mal_studies = [
            optuna.create_study(
                directions=["maximize", "maximize"],  # Adversary wants to maximize BDA and MTA
                sampler=NSGAIISampler(constraints_func=mta_constraint, population_size=args.population_size_malicious),
            ) if mta_constrained else optuna.create_study(
                directions=["maximize", "maximize"],  # Adversary wants to maximize BDA and MTA
                sampler=NSGAIISampler(population_size=args.population_size_malicious),
            ) for _ in range(len(benign_configurations))
        ]

        for gen_mal in range(args.generations_malicious):
            logging.info(f'Starting malicious generation {gen_mal} / {args.generations_malicious}')
            mal_studies_trials = [
                [] for _ in range(len(benign_configurations))
            ]
            configurations = []

            for i, study in enumerate(mal_studies):
                for _ in range(args.population_size_malicious):
                    trial = study.ask()
                    config = {
                        "beta": trial.suggest_float(name="beta", low=0.2, high=10),
                        "wd_m": trial.suggest_float(name="wd_m", low=0.0001, high=0.001),
                        "E_m": trial.suggest_int(name="E_m", low=2, high=20),
                        "B_m": trial.suggest_int(name="B_m", low=16, high=128, step=16),
                    }
                    config.update(benign_configurations[i])
                    experiment_str = f"{args.experiment_descriptor_prefix}_{config['lr_b']}_{config['wd_b']}_{config['E_b']}_{config['B_b']}/{config['beta']}_{config['wd_m']}_{config['E_m']}_{config['B_m']}"
                    already_tested = False
                    for k_ben, malicious_generations in generations_to_configurations.items():
                        for k_mal, experiments in malicious_generations.items():
                            if experiment_str in experiments:
                                already_tested = True
                                logging.info(f'Config {config} in {gen_mal=} was already tested in generation {k_ben} / {k_mal}.')
                    config['already_tested'] = already_tested

                    mal_studies_trials[i].append(trial)
                    configurations.append(config)

            logging.info(f'Malicious configurations in benign generation {gen_ben} and malicious generation {gen_mal} are {configurations}')

            generations_to_configurations[gen_ben][gen_mal] = [f"{args.experiment_descriptor_prefix}_{config['lr_b']}_{config['wd_b']}_{config['E_b']}_{config['B_b']}/{config['beta']}_{config['wd_m']}_{config['E_m']}_{config['B_m']}" for config in configurations]

            with open(f"logs/{args.experiment_descriptor_prefix}/generations_to_configurations.json", "w") as f:
                json.dump(generations_to_configurations, f)

            objectives = run_batch(configurations)

            # finish all malicious trials
            for i, (study, mal_trials) in enumerate(zip(mal_studies, mal_studies_trials)):
                for j, trial in enumerate(mal_trials):
                    mta, bda = objectives[i * args.population_size_malicious + j]
                    logging.debug(f'In {gen_ben=} for config {i} in {gen_mal=} {trial.number=} yielded MTA: {mta} BDA: {bda}')
                    trial.set_user_attr("MTA", mta)
                    study.tell(trial, (mta, bda))

        mtas = []
        bdas = []

        for i, study in enumerate(mal_studies):
            pareto_trials = study.best_trials
            if len(pareto_trials) == 0: # No trial fulfilled the constraint or something crashed
                logging.warning(f'There was no valid malicious solution for config {i} in {gen_ben=}, resorting to reporting trial with highest MTA')
                pareto_trials = [max(study.trials, key=lambda t: t.values[0])]
            logging.debug(f'Best adversary for configuration {i} achieved (MTA/BDA) = {max(pareto_trials, key=lambda x: x.values[1]).values}')
            ben_objectives.append(max(pareto_trials, key=lambda x: x.values[1]).values)

        # finish all trials in the batch
        for trial, objective in zip(ben_trials, ben_objectives):
            trial.set_user_attr("MTA", objective[0])
            ben_study.tell(trial, objective)

    # Access the Pareto front
    pareto_trials = ben_study.best_trials
    for t in pareto_trials:
        logging.info(f"Params: {t.params}, MTA: {t.values[0]}, BDA: {t.values[1]}")
