import json
import os
from operator import indexOf

import numpy as np
import yaml

from evaluation.hyper_params.utils import plot_graphs, extract_lifespans, extract_accuracies

if __name__=="__main__":
    benign = [0.8, 0.9, 0.99]
    malicious = [0.8, 0.9, 0.99]
    attacks = ['A3FL', 'Chameleon', 'IBA', 'DataPoisoning', 'DarkFed', 'FCBA']

    results_bda_during = {}
    results_bda_after = {}
    results_mta_during = {}
    results_mta_after = {}
    results_bda_auc = {}
    results_50_lifespan = {}

    mta_during = ""
    bda_during = ""
    mta_after = ""
    bda_after = ""

    lifespan_50 = ""
    lifespan_mean = ""
    lifespan_weighted_mean = ""

    auc_during_after = ""

    for attack in attacks:
        results_bda_during[attack] = np.zeros((len(malicious), len(benign)))
        results_bda_after[attack] = np.zeros((len(malicious), len(benign)))
        results_mta_during[attack] = np.zeros((len(malicious), len(benign)))
        results_mta_after[attack] = np.zeros((len(malicious), len(benign)))
        results_bda_auc[attack] = np.zeros((len(malicious), len(benign)))
        results_50_lifespan[attack] = np.zeros((len(malicious), len(benign)))

        for i, mal in enumerate(malicious):
            for j, ben in enumerate(benign):
                dir = f'logs/hyper_params/momentum/{attack}_{ben}_{mal}'

                results = extract_accuracies(dir)

                mta_during += f"{round(100 * results['during_attack_window']['MTA'], 2)} %\t"
                bda_during += f"{round(100 * results['during_attack_window']['BDA'], 2)} %\t"
                mta_after += f"{round(100 * results['after_attack_window']['MTA'], 2)} %\t"
                bda_after += f"{round(100 * results['after_attack_window']['BDA'], 2)} %\t"

                results_bda_during[attack][i, j] = results['during_attack_window']['BDA']
                results_bda_after[attack][i, j] = results['after_attack_window']['BDA']
                results_mta_during[attack][i, j] = results['during_attack_window']['MTA']
                results_mta_after[attack][i, j] = results['after_attack_window']['MTA']

                l_50, l_m, l_wm, auc_results = extract_lifespans(dir)
                lifespan_50 += l_50
                lifespan_mean += l_m
                lifespan_weighted_mean += l_wm
                auc_during_after += f"{auc_results['during & after_attack_window']['AUC_BDA']} \t"
                results_bda_auc[attack][i, j] = auc_results['during & after_attack_window']['AUC_BDA']
                results_50_lifespan[attack][i, j] = l_50

            mta_during += "\n"
            bda_during += "\n"
            mta_after += "\n"
            bda_after += "\n"
            lifespan_50 += "\n"
            lifespan_mean += "\n"
            lifespan_weighted_mean += "\n"
            auc_during_after += "\n"

        if os.path.exists(f'logs/Original_{attack}_uniform_model'):
            temp = extract_accuracies(f'logs/Original_{attack}_uniform_model')
            l_50, _, _, auc_temp = extract_lifespans(f'logs/Original_{attack}_uniform_model')
            with open(f'logs/Original_{attack}_uniform_model/config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            original_vals = {
                'mal_value': config['adversarial_optimizer']['momentum'],
                'ben_value': config['optimizer']['momentum'],
                'bda_during': round(100 * temp['during_attack_window']['BDA'], 2),
                'mta_during': round(100 * temp['during_attack_window']['MTA'], 2),
                'bda_after': round(100 * temp['after_attack_window']['BDA'], 2),
                'mta_after': round(100 * temp['after_attack_window']['MTA'], 2),
                'bda_auc_during_after': auc_temp['during & after_attack_window']['AUC_BDA'],
                '50_lifespan': int(l_50),
            }
        else:
            original_vals = None

        plot_graphs(malicious, benign, results_bda_during[attack], results_bda_after[attack], results_bda_auc[attack],
                    results_mta_during[attack], results_mta_after[attack], results_50_lifespan[attack], r'\mu_m', r'\mu_b',
                    f'Momentum - {attack}')

    print("MTA during:\n")
    print(mta_during)
    print("BDA during:\n")
    print(bda_during)
    print("MTA after:\n")
    print(mta_after)
    print("BDA after:\n")
    print(bda_after)
    print("Lifespan 50%:\n")
    print(lifespan_50)
    print("Lifespan mean:\n")
    print(lifespan_mean)
    print("Lifespan weighted mean:\n")
    print(lifespan_weighted_mean)
    print("BDA AUC after:\n")
    print(auc_during_after)

    best_choices = {}
    for attack in attacks:
        attack_snake_case = "".join(["_" + c.lower() if c.isupper() else c.lower() for c in attack])
        if attack_snake_case[0] == "_":
            attack_snake_case = attack_snake_case[1:]
        best_choices[attack] = {}
        for ben in benign:
            best_index = np.argmax(results_bda_during[attack][:, indexOf(benign, ben)])
            best_choices[attack][ben] = malicious[best_index]

    with open('./evaluation/hyper_params/results/best_mal_choices_greedy_mu.json', 'w') as f:
        json.dump(best_choices, f)

    balancing_param = 0.5
    best_choices = {}
    for attack in attacks:
        attack_snake_case = "".join(["_" + c.lower() if c.isupper() else c.lower() for c in attack])
        if attack_snake_case[0] == "_":
            attack_snake_case = attack_snake_case[1:]
        best_choices[attack] = {}
        for ben in benign:
            total_acc = (1 - balancing_param) * results_bda_during[attack][:, indexOf(benign, ben)] + balancing_param * results_mta_during[attack][:, indexOf(benign, ben)]
            best_index = np.argmax(total_acc)
            best_choices[attack][ben] = malicious[best_index]

    with open('./evaluation/hyper_params/results/best_mal_choices_balanced_mu.json', 'w') as f:
        json.dump(best_choices, f)

    allowed_mta_decreases = [0.02, 0.05]
    for allowed_mta_decrease in allowed_mta_decreases:
        best_choices = {}
        for attack in attacks:
            attack_snake_case = "".join(["_" + c.lower() if c.isupper() else c.lower() for c in attack])
            if attack_snake_case[0] == "_":
                attack_snake_case = attack_snake_case[1:]
            best_choices[attack] = {}
            for ben in benign:
                mtas = results_mta_during[attack][:, indexOf(benign, ben)]
                bdas = results_bda_during[attack][:, indexOf(benign, ben)]
                max_mta = mtas.max()
                mta_constraint = max_mta - allowed_mta_decrease

                # Get mask of indices that satisfy the MTA constraint
                valid_indices = np.nonzero(mtas >= mta_constraint)[0]
                # Among valid indices, select the one with the highest BDA
                selected_index = valid_indices[np.argmax(bdas[valid_indices])]

                best_choices[attack][ben] = malicious[selected_index]

        with open(f'./evaluation/hyper_params/results/best_mal_choices_MTA_constrained_{allowed_mta_decrease}_mu.json', 'w') as f:
            json.dump(best_choices, f)
