import itertools
import os

import numpy as np
import yaml
from matplotlib import pyplot as plt

from evaluation.hyper_params.utils import extract_accuracies, extract_lifespans

def get_pareto_front(criteria1, sort1, criteria2, sort2):
    front = []
    for i in range(len(criteria1)):
        c1, c2 = criteria1[i], criteria2[i]
        front = [j for j in front if ((criteria1[j] > c1 and sort1 == 'asc') or (criteria1[j] < c1 and sort1 == 'desc')) or ((criteria2[j] > c2 and sort2 == 'asc') or (criteria2[j] < c2 and sort2 == 'desc'))]
        dominated = False
        for j in front:
            x1, x2 = criteria1[j], criteria2[j]
            if ((x1 >= c1 and sort1 == 'asc') or (x1 <= c1 and sort1 == 'desc')) and ((x2 >= c2 and sort2 == 'asc') or (x2 <= c2 and sort2 == 'desc')):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front

def get_config(experiment):
    with open(f'{experiments_path}/{experiment}/config.yaml') as f:
        return yaml.safe_load(f)

def get_parameter_string(configuration):
    lr_b = configuration['optimizer']['local_lr']
    mu_b = configuration['optimizer']['momentum']
    wd_b = configuration['optimizer']['weight_decay']
    E_b = configuration['local_epochs']
    B_b = configuration['batch_size']
    return f"{lr_b} | {mu_b} | {wd_b} | {E_b} | {B_b}"


if __name__=="__main__":
    attacks = ['A3FL', 'Chameleon', 'DarkFed', 'FCBA']
    defenses = ['Bulyan', 'FoolsGold', 'Krum', 'MultiKrum', 'None']

    experiments_path = './logs/hyper_params/pareto'

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 15
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'

    most_frequent_hyperparameters = {}

    for attack in attacks:
        experiments = [ f.name for f in os.scandir(experiments_path) if f.is_dir()  and attack.lower() in f.name.lower() ]
        mtas_during = {}
        bdas_during = {}
        baccs_during = {}
        mtas_after = {}
        bdas_after = {}
        baccs_after = {}
        mta_aucs_during_after = {}
        bda_aucs_during_after = {}
        lifespans_50 = {}

        for experiment in experiments:
            dir = f'{experiments_path}/{experiment}'

            if os.path.exists(f'{dir}/average_accs.log'):
                defense = experiment.split('_')[1]
                if defense not in mtas_during.keys():
                    mtas_during[defense] = []
                    bdas_during[defense] = []
                    baccs_during[defense] = []
                    mtas_after[defense] = []
                    bdas_after[defense] = []
                    baccs_after[defense] = []
                    mta_aucs_during_after[defense] = []
                    bda_aucs_during_after[defense] = []
                    lifespans_50[defense] = []

                results = extract_accuracies(dir)

                mtas_during[defense].append(results['during_attack_window']['MTA'])
                bdas_during[defense].append(results['during_attack_window']['BDA'])
                baccs_during[defense].append(results['during_attack_window']['BACC'])
                mtas_after[defense].append(results['after_attack_window']['MTA'])
                bdas_after[defense].append(results['after_attack_window']['BDA'])
                baccs_after[defense].append(results['after_attack_window']['BACC'])

                lifespan_50, _, _, auc_results = extract_lifespans(dir)
                mta_aucs_during_after[defense].append(auc_results['during & after_attack_window']['AUC_MTA'])
                bda_aucs_during_after[defense].append(auc_results['during & after_attack_window']['AUC_BDA'])
                lifespans_50[defense].append(int(lifespan_50))

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Improve readability in black-and-white
        tmp = colors[4]
        colors[4] = colors[1]
        colors[1] = tmp

        linestyles = ['-', '--', '-.', ':']

        # Generate Pareto Plot for BDA during vs. MTA during
        fig, ax = plt.subplots()

        for j, defense in enumerate(defenses):
            front = get_pareto_front(bdas_during[defense], 'desc', mtas_during[defense], 'asc')
            print(f'Pareto Front for {attack} vs. {defense} (BDA during vs MTA during):')

            print('| | $\\eta_b$ | $\\mu_b$ | $\\lambda_b$ | $E_b$ | $B_b$ | $\\text{BDA}_\\text{during}$ | $\\text{MTA}_\\text{during}$ |', "\n", "| --- " * 8, "|")
            defense_experiments = [experiment for experiment in experiments if f'_{defense}_' in experiment]
            for i in front:
                mta_during = mtas_during[defense][i]
                bda_during = bdas_during[defense][i]
                print(f'| {defense_experiments[i]} | {get_parameter_string(get_config(defense_experiments[i]))} | {round(bda_during, 2)} | {round(mta_during, 2)} |')

                if (key := "_".join(defense_experiments[i].split('_')[2:])) in most_frequent_hyperparameters:
                    most_frequent_hyperparameters[key]['frequency'] += 1
                else:
                    most_frequent_hyperparameters[key] = {
                        'frequency': 1,
                    }

            arr1, arr2 = np.array(bdas_during[defense])[front], np.array(mtas_during[defense])[front]
            sort_indices = np.argsort(arr1)
            arr1, arr2 = arr1[sort_indices], arr2[sort_indices]

            ax.plot(100 * arr1, 100 * arr2, color=colors[j], linestyle=linestyles[j % len(linestyles)], label=defense)

            # Plot the Pareto front points
            ax.scatter(100 * arr1, 100 * arr2, color=colors[j], zorder=3)

        plt.xlabel(r'$\mathtt{BDA}$ [\%]')
        plt.ylabel(r'$\mathtt{MTA}$ [\%]')
        plt.grid(True)

        os.makedirs('evaluation/pareto_front/results', exist_ok=True)

        plt.legend()
        plt.tight_layout()
        plt.savefig(f'evaluation/pareto_front/results/Pareto_{attack}_Accs_During.pdf')
        plt.show()

        # Generate Pareto Plot for BDA after vs. MTA after
        fig, ax = plt.subplots()

        for j, defense in enumerate(defenses):
            front = get_pareto_front(bdas_after[defense], 'desc', mtas_after[defense], 'asc')
            print(f'Pareto Front for {attack} (BDA after vs MTA after):')

            defense_experiments = [experiment for experiment in experiments if f'_{defense}_' in experiment]
            for i in front:
                mta_auc = mtas_after[defense][i]
                bda_auc = bdas_after[defense][i]
                print(f'\t{i} => {defense_experiments[i]} ({get_parameter_string(get_config(defense_experiments[i]))}) => avg. MTA AuC: {round(mta_auc * 100, 2)} %, AuC: {round(bda_auc, 2)}')
            arr1, arr2 = np.array(bdas_after[defense])[front], np.array(mtas_after[defense])[front]
            sort_indices = np.argsort(arr1)
            arr1, arr2 = arr1[sort_indices], arr2[sort_indices]

            ax.plot(100 * arr1, 100 * arr2, color=colors[j], linestyle=linestyles[j % len(linestyles)], label=defense)

            # Plot the Pareto front points
            ax.scatter(100 * arr1, 100 * arr2, color=colors[j], zorder=3)

        plt.xlabel(r'$\mathtt{BDA^*}$ [\%]')
        plt.ylabel(r'$\mathtt{MTA^*}$ [\%]')
        plt.grid(True)

        plt.legend()
        plt.tight_layout()
        plt.savefig(f'evaluation/pareto_front/results/Pareto_{attack}_Accs_After.pdf')
        plt.show()

    for hyperparam in most_frequent_hyperparameters:
        most_frequent_hyperparameters[hyperparam].update({
            'mtas': np.zeros(len(attacks) * len(defenses)),
            'bdas': np.zeros(len(attacks) * len(defenses)),
        })

        for i, defense in enumerate(defenses):
            for j, attack in enumerate(attacks):
                dir = f'{experiments_path}/{attack}_{defense}_{hyperparam}'

                if os.path.exists(f'{dir}/average_accs.log'):
                    results = extract_accuracies(dir)
                    mta = results['during_attack_window']['MTA']
                    bda = results['during_attack_window']['BDA']

                    most_frequent_hyperparameters[hyperparam]['mtas'][i * len(attacks) + j] = mta
                    most_frequent_hyperparameters[hyperparam]['bdas'][i * len(attacks) + j] = bda

    # Filter out entries where average BDA is >= 20%
    filtered = {k: v for k, v in most_frequent_hyperparameters.items() if np.mean(v['bdas']) < 0.2}

    # Sort the filtered dict by frequency (descending)
    sorted_filtered = dict(sorted(filtered.items(), key=lambda item: item[1]['frequency'], reverse=True))

    best_config, best_values = next(iter(sorted_filtered.items()))
    print(f'The best parameters are: {best_config} => Avg. MTA / BDA = ({round(100 * np.mean(best_values["mtas"]), 2):.1f} % / {round(100 * np.mean(best_values["bdas"]), 2):.1f} %)')

    for config, item in most_frequent_hyperparameters.items():
        print(f'{config} => Frequency: {item["frequency"]}, Avg. MTA: {np.mean(item["mtas"])}, Avg. BDA: {np.mean(item["bdas"])}')
