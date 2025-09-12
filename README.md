# On Hyperparameters and Backdoor-Resistance in Horizontal Federated Learning

This repository contains the Python source code and the empirical results for the experiments of the paper "On Hyperparameters and Backdoor-Resistance in Horizontal Federated Learning", published in the Proceedings of the ACM Conference on Computer and Communications Security (CCS) 2025.

- [PrePrint on ArXiv](https://arxiv.org/abs/2509.05192)
- _Published Conference Paper (TBD)_

## Setup

The code was tested with Python 3.10.12 using a Conda environment as defined in `conda_env.yaml`.

### Installation

1. Clone the repository
2. [Install Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
3. Create the conda environment
```bash
conda env create -f conda_env.yaml
```
4. Activate the conda environment using `conda activate artifact`

## Usage

The framework supports four ways to start experiments, each documented in a separate subsection in the following.

### 1. Starting a single experiment using `FLSimulation.py`

The most basic way to start an experiment is to create a config `config.yaml` file and run the simulation defined within using

```
python FLSimulation.py --params config.yaml
```

You can take inspiration on possible configuration options from the `./experiments` folder.

`FLSimulation.py` supports several command line arguments which can be listed using the `--help` flag:

```bash
python FLSimulation.py --help
usage: FLSimulation.py [-h] [--params PARAMS] [--experiment_descriptor EXPERIMENT_DESCRIPTOR] [--reproducible] [--seed SEED] [--devices DEVICES] [--num_workers NUM_WORKERS] [--visdom_port VISDOM_PORT]
                       [--test_batch_size TEST_BATCH_SIZE] [-y] [--use_visdom]

Backdoor EvAluation Framework (BEAF)

options:
  -h, --help            show this help message and exit
  --params PARAMS       Path to YAML config file
  --experiment_descriptor EXPERIMENT_DESCRIPTOR
                        A unique name for the experiment. If not set, an automatic name is generated. This also defines the subfolder inside ./logs where the experiment will be logged.
  --reproducible        Run the experiments in reproducible mode by seeding all RNGs. May impact performance.
  --seed SEED           A seed for all random number generators.
  --devices DEVICES     A comma separated list of device indices to run the experiment on. Taken from config file, if not specified.
  --num_workers NUM_WORKERS
                        The number of CPU processes to use for the DataLoader. Keep in mind, that this number of processes is spawned per active client! Depending on the attack and backdoor type used, this can cause the
                        client_communicator actor and, therefore, the whole experiment to fail.
  --visdom_port VISDOM_PORT
                        The port number for the visdom server instance to connect to.
  --test_batch_size TEST_BATCH_SIZE
                        The test batch size to use.
  -y                    Silently confirm all warnings.
  --use_visdom          Visualize results using visdom.
```

#### Visualization

When using `FLSimulation.py`, running experiments can be visualized using [visdom](https://pypi.org/project/visdom/).
To this end, start a visdom server in a long-running background task using
```bash
python -m visdom.server -port <PORT>
```
Use the argument `--use_visdom` and specify the selected port (default is 8097) in the `--visdom_port` argument and connect to the server in your browser.

### 2. Starting multiple experiments using `multi_run.py`

`multi_run.py` can be used to start a grid search over several variations of a base configuration automatically. 
It creates every configuration as a job and schedules job based on GPU-availability.

It supports several command line parameters:

```bash
python multi_run.py --help
usage: multi_run.py [-h] --base_config BASE_CONFIG [--gpus_per_client GPUS_PER_CLIENT] [--cpus_per_client CPUS_PER_CLIENT] [--experiment_descriptor_prefix EXPERIMENT_DESCRIPTOR_PREFIX]
                    [--experiment_descriptor_template EXPERIMENT_DESCRIPTOR_TEMPLATE] --devices DEVICES [DEVICES ...] --tasks_per_gpu TASKS_PER_GPU [TASKS_PER_GPU ...] [--num_workers NUM_WORKERS] [--visdom_port VISDOM_PORT]
                    [--use_visdom] [--test_batch_size TEST_BATCH_SIZE] [--malicious_strategy {greedy,balanced,MTA_constrained_0.02,MTA_constrained_0.05}] [--seed SEED [SEED ...]] [--test]

FLSimulation Multi Run

options:
  -h, --help            show this help message and exit
  --base_config BASE_CONFIG
                        Path to base config file
  --gpus_per_client GPUS_PER_CLIENT
                        Number of GPUs required per FL client
  --cpus_per_client CPUS_PER_CLIENT
                        Number of CPUs required per FL client
  --experiment_descriptor_prefix EXPERIMENT_DESCRIPTOR_PREFIX
                        Prefix to add to all experiment descriptors - rest of the experiment descriptor will simply count upwards
  --experiment_descriptor_template EXPERIMENT_DESCRIPTOR_TEMPLATE
                        Template for experiment descriptors. You can use every variable from the configuration file as a placeholder.
  --devices DEVICES [DEVICES ...]
                        Space-separated list of GPU indices to use
  --tasks_per_gpu TASKS_PER_GPU [TASKS_PER_GPU ...]
                        Space-separated list of tasks per GPU (must have same length as --devices)
  --num_workers NUM_WORKERS
                        Number of dataloader workers
  --visdom_port VISDOM_PORT
                        Port on which the visdom server runs
  --use_visdom           Visualize results using visdom.
  --test_batch_size TEST_BATCH_SIZE
                        The test batch size to use.
  --malicious_strategy {greedy,balanced,MTA_constrained_0.02,MTA_constrained_0.05}
                        The strategy that is used to determine the malicious hyperparameters
  --seed SEED [SEED ...]
                        A seed for all random number generators.
```

Note that the `--experiment_descriptor_prefix` and `--experiment_descriptor_template` are mutually exclusive.

The `--malicious_strategy` you specify is used whenever you specify a hyperparameter to be '?' (see "Specifying the search space").
In this case, it loads the corresponding "ideal" malicious parameter from `evaluation/hyper_params/results/best_mal_choices_*.json`.
Note that this works only for the learning rate, momentum, weight decay, batch size and number of local epochs.

#### Specifying the search space

To specify the search space of the grid search, you can specify any config file key (nested keys are concatenated using `/`) and specify one of the following for it:

1. A single value: This value will be set for the corresponding key.
2. A space-separated list of values: For each value, a single experiment will be started.
3. `.`: The value will be used as specified in the base config file
4. `?`: The value will be chosen according to the selected `--malicious_strategy` (only works for `beta`, `mu_m`, `wd_m`, `E_m`, `B_m`)

Therefore, the following command line spawns in total 24 experiments distributed across three GPUs from which each can run 2 experiments simultaneously, resulting in a total run time of roughly 4x the run time of a single experiment:

```bash
python multi_run.py --base_config experiments/hyper_params/attacks_vs_defenses_pareto_short/A3FL_Bulyan.yaml \
                    --lr_b 0.1 0.15 0.2 \
                    --mu_b 0.9 \
                    --wd_b 0.001 \
                    --E_b 10 20 \
                    --B_b 16 32 \
                    --beta ? \
                    --wd_m ? \
                    --E_m ? \
                    --B_m ? \
                    --experiment_descriptor_template hyper_params/pareto/A3FL_Bulyan_{lr_b}_{wd_b}_{E_b}_{B_b} \
                    --devices 0 1 2 \
                    --tasks_per_gpu 2 2 2 \
                    --test_batch_size 128 \
                    --num_workers 0 \
                    --malicious_strategy greedy
```

When you instead of a full grid search want to run a set of specific configurations, there is another syntax to use:
You can chain parameters (those that correspond to configuration file values) using `&` signs (note that you might need to escape the `&` sign in bash), allowing to specify lists of tuples that are to be evaluated.
For example, the following command line evaluates the base config file with (`E_b = 5`, `B_b = 16`) and (`E_b = 10`, `B_b = 32`) but not the missing combinations:

```bash
python multi_run.py --base_config experiments/hyper_params/attacks_vs_defenses_pareto_short/A3FL_Bulyan.yaml \
                    --E_b\&B_b 5 16 10 32 \
                    --experiment_descriptor_template hyper_params/pareto/A3FL_Bulyan_{lr_b}_{wd_b}_{E_b}_{B_b}  \
                    --devices 0  \
                    --tasks_per_gpu 2 \
                    --test_batch_size 128  \
                    --num_workers 0 
```

### 3. Automated grid search using NSGA-II

There are two runner scripts that support the use of the genetic optimization algorithm [NSGA-II](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.NSGAIISampler.html) to search the parameter space.
Note that these scripts are not general purpose but each created for a very specific purpose.

#### 3.1 Using `multi_run_NSGA-II.py` for an approximation of the Pareto frontier

The `multi_run_NSGA-II.py` script can be used to approximate the pareto frontier w.r.t. main task and backdoor accuracy during the attack window when both the benign and malicious clients fine-tune their ML training hyperparameters (learning rate, weight decay, number of local epochs, batch size) using NSGA-II.
It supports the following command line arguments:

```bash
python multi_run_NSGA-II.py --help
usage: multi_run_NSGA-II.py [-h] [--experiment_descriptor_prefix EXPERIMENT_DESCRIPTOR_PREFIX] --base_config BASE_CONFIG [--gpus_per_client GPUS_PER_CLIENT] [--cpus_per_client CPUS_PER_CLIENT] --devices DEVICES [DEVICES ...]
                            --tasks_per_gpu TASKS_PER_GPU [TASKS_PER_GPU ...] [--num_workers NUM_WORKERS] [--test_batch_size TEST_BATCH_SIZE] [--population_size_benign POPULATION_SIZE_BENIGN]
                            [--population_size_malicious POPULATION_SIZE_MALICIOUS] [--generations_benign GENERATIONS_BENIGN] [--generations_malicious GENERATIONS_MALICIOUS] [--ideal_mta IDEAL_MTA] [--mta_diff MTA_DIFF]
                            [--reproducible]

Approximate the Pareto front using NSGA-II

options:
  -h, --help            show this help message and exit
  --experiment_descriptor_prefix EXPERIMENT_DESCRIPTOR_PREFIX
                        Prefix for all experiment descriptors. Remaining part will be auto-generated.
  --base_config BASE_CONFIG
                        Base config file
  --gpus_per_client GPUS_PER_CLIENT
                        Number of GPUs required per FL client
  --cpus_per_client CPUS_PER_CLIENT
                        Number of CPUs required per FL client
  --devices DEVICES [DEVICES ...]
                        Space-separated list of GPU indices to use
  --tasks_per_gpu TASKS_PER_GPU [TASKS_PER_GPU ...]
                        Space-separated list of tasks per GPU (must have same length as --devices)
  --num_workers NUM_WORKERS
                        Number of dataloader workers
  --test_batch_size TEST_BATCH_SIZE
                        The test batch size to use
  --population_size_benign POPULATION_SIZE_BENIGN
                        The number of individuals to create in each generation of the benign NSGA-II optimization run
  --population_size_malicious POPULATION_SIZE_MALICIOUS
                        The number of individuals to create in each generation of the malicious NSGA-II optimization run
  --generations_benign GENERATIONS_BENIGN
                        The number of generations to run the benign NSGA-II optimization for
  --generations_malicious GENERATIONS_MALICIOUS
                        The number of generations to run the malicious NSGA-II optimization for
  --ideal_mta IDEAL_MTA
                        The maximum MTA an undefended and unattacked model achieves
  --mta_diff MTA_DIFF   The maximum allowed decrease in MTA caused by an attack or defense. Do not set to enable unconstrained optimization.
  --reproducible        Fix all random seeds
```

#### 3.2 Using `multi_run_NSGA-II_malicious.py` to emulate a strong adversary

The `multi_run_NSGA-II_malicious.py` script allows to emulate an adversary that uses NSGA-II to tweak its ML training hyperparameters (learning rate, weight decay, number of local epochs, batch size) to launch stronger attacks against a fixed set of benign hyperparameters.
It supports the following command line arguments:

```bash
python multi_run_NSGA-II_malicious.py --help
usage: multi_run_NSGA-II_malicious.py [-h] [--experiment_descriptor_prefix EXPERIMENT_DESCRIPTOR_PREFIX] --base_config BASE_CONFIG [--gpus_per_client GPUS_PER_CLIENT] [--cpus_per_client CPUS_PER_CLIENT] --devices DEVICES
                                      [DEVICES ...] --tasks_per_gpu TASKS_PER_GPU [TASKS_PER_GPU ...] [--num_workers NUM_WORKERS] [--test_batch_size TEST_BATCH_SIZE] [--population_size_malicious POPULATION_SIZE_MALICIOUS]
                                      [--generations_malicious GENERATIONS_MALICIOUS] [--ideal_mta IDEAL_MTA] [--mta_diff MTA_DIFF] [--reproducible] [--lr_b LR_B] [--wd_b WD_B] [--E_b E_B] [--B_b B_B]

Approximate the Pareto front using NSGA-II

options:
  -h, --help            show this help message and exit
  --experiment_descriptor_prefix EXPERIMENT_DESCRIPTOR_PREFIX
                        Prefix for all experiment descriptors. Remaining part will be auto-generated.
  --base_config BASE_CONFIG
                        Base config file
  --gpus_per_client GPUS_PER_CLIENT
                        Number of GPUs required per FL client
  --cpus_per_client CPUS_PER_CLIENT
                        Number of CPUs required per FL client
  --devices DEVICES [DEVICES ...]
                        Space-separated list of GPU indices to use
  --tasks_per_gpu TASKS_PER_GPU [TASKS_PER_GPU ...]
                        Space-separated list of tasks per GPU (must have same length as --devices)
  --num_workers NUM_WORKERS
                        Number of dataloader workers
  --test_batch_size TEST_BATCH_SIZE
                        The test batch size to use
  --population_size_malicious POPULATION_SIZE_MALICIOUS
                        The number of individuals to create in each generation of the malicious NSGA-II optimization run
  --generations_malicious GENERATIONS_MALICIOUS
                        The number of generations to run the malicious NSGA-II optimization for
  --ideal_mta IDEAL_MTA
                        The maximum MTA an undefended and unattacked model achieves
  --mta_diff MTA_DIFF   The maximum allowed decrease in MTA caused by the attack. Do not set to enable unconstrained optimization.
  --reproducible        Fix all random seeds
  --lr_b LR_B           The benign learning rate
  --wd_b WD_B           The benign weight decay
  --E_b E_B             The benign local epochs
  --B_b B_B             The benign batch size
  ```

## Evaluation of Results

The log directory of a completed experiment contains six files:

```bash
average_accs.log            // The average MTA and BDA before, during, and after the attack window
config.yaml                 // A copy of the configuration file with which the experiment was started
data_distribution.png       // A visualization of the data distribution across all FL clients
history.csv                 // A CSV file containing all MTAs and BDAs in every round
history_intermediate.csv    // Same as history.csv but already available and continuously updated during a run
lifespans.log               // An evaluation of the lifespan the attack achieved
log.log                     // The full log. Worker failures can be detected here
```

## Paper results

To ensure reproducibility of our results despite high computational requirements of the project, we provide the log files for all conducted experiments.
These are located in `logs/Experiments_CCS_2025_share.tar.bz2`.
To reprooduce the plots and tables from the paper:

1. Unzip the experimental data: `tar -xvjf Experiments_CCS_2025_share.tar.bz2`
2. Move and rename the corresponding folder to `logs/hyper_params`: `mv Experiments_CCS_2025_share logs/hyper_params`
3. From the top-level folder, run the corresponding evaluation script from `./evaluation`. If you encounter import errors, make sure to always prepend the python command `PYTHONPATH=$(pwd)`.

### Supported Experiments

The following evaluation scripts are supported:

| Name                                                                | Description                                          |
|---------------------------------------------------------------------|------------------------------------------------------|
| `evaluation/hyper_params/Evaluate_per_parameter_greedy_strategy.py` | Generate Table VI                                    |
| `evaluation/hyper_params/Evaluate_B.py`                             | Plots for the benign/malicious batch size ablation   |
| `evaluation/hyper_params/Evaluate_E.py`                             | Plots for the benign/malicious local epochs ablation |
| `evaluation/hyper_params/Evaluate_lr.py`                            | Plots for the benign/malicious learning rate ablation |
| `evaluation/hyper_params/Evaluate_momentum.py`                      | Plots for the benign/malicious momentum ablation     |
| `evaluation/hyper_params/Evaluate_wd.py`                            | Plots for the benign/malicious weight decay ablation |
| `evaluation/hyper_params/Multiple_Linear_Regression.py`             | Regression analysis (Table III)                      |
| `evaluation/pareto_front/Evaluation_pareto.py`                      | Plot the Pareto frontiers (Figure 10)                |
| `evaluation/pareto_front/MTA_before_evaluation.py`                  | Compare MTA before the attack window                 |
| `evaluation/pareto_front/Pareto_at_point_of_operation.py`           | Generate Table V                                     |

## Additional Notes

Please note that the DarkFed and FCBA attacks (both used in the paper) are not currently supported in this codebase due to ongoing licensing considerations.
We plan to make them available in the future.

## Citation

_Will be added, once the proceedings of the ACM CCS 2025 conference are published._

## Contact

Feel free to contact the first author via the e-mail provided on the publication if you have any questions.
