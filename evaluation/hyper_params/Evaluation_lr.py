import json
import os
from operator import indexOf

import numpy as np
import yaml

from evaluation.hyper_params.utils import plot_graphs, extract_accuracies, extract_lifespans

if __name__=="__main__":
    benign = [0.05, 0.1, 0.2, 0.5]
    malicious = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    attacks = ['A3FL', 'Chameleon', 'IBA', 'DataPoisoning', 'DarkFed', 'FCBA']

    results_bda_during = {}
    results_bda_after = {}
    results_mta_during = {}
    results_mta_after = {}
    results_bda_auc = {}
    results_50_lifespan = {}

    for schedule in ["uniform", "decay"]:
        mta_during = ""
        bda_during = ""
        mta_after = ""
        bda_after = ""

        lifespan_50 = ""
        lifespan_mean = ""
        lifespan_weighted_mean = ""

        auc_during_after = ""

        results_bda_during[schedule] = {}
        results_bda_after[schedule] = {}
        results_mta_during[schedule] = {}
        results_mta_after[schedule] = {}
        results_bda_auc[schedule] = {}
        results_50_lifespan[schedule] = {}

        print(f"=== Schedule: {schedule} ===")

        for attack in attacks:
            results_bda_during[schedule][attack] = np.zeros((len(malicious), len(benign)))
            results_bda_after[schedule][attack] = np.zeros((len(malicious), len(benign)))
            results_mta_during[schedule][attack] = np.zeros((len(malicious), len(benign)))
            results_mta_after[schedule][attack] = np.zeros((len(malicious), len(benign)))
            results_bda_auc[schedule][attack] = np.zeros((len(malicious), len(benign)))
            results_50_lifespan[schedule][attack] = np.zeros((len(malicious), len(benign)))

            for i, mal in enumerate(malicious):
                for j, ben in enumerate(benign):
                    dir = f'logs/hyper_params/learning_rate/{attack}_{ben}_{mal}_{schedule}'

                    results = extract_accuracies(dir)

                    mta_during += f"{round(100 * results['during_attack_window']['MTA'], 2)} %\t"
                    bda_during += f"{round(100 * results['during_attack_window']['BDA'], 2)} %\t"
                    mta_after += f"{round(100 * results['after_attack_window']['MTA'], 2)} %\t"
                    bda_after += f"{round(100 * results['after_attack_window']['BDA'], 2)} %\t"

                    results_bda_during[schedule][attack][i, j] = results['during_attack_window']['BDA']
                    results_bda_after[schedule][attack][i, j] = results['after_attack_window']['BDA']
                    results_mta_during[schedule][attack][i, j] = results['during_attack_window']['MTA']
                    results_mta_after[schedule][attack][i, j] = results['after_attack_window']['MTA']

                    l_50, l_m, l_wm, auc_results = extract_lifespans(dir)
                    lifespan_50 += l_50
                    lifespan_mean += l_m
                    lifespan_weighted_mean += l_wm
                    auc_during_after += f"{auc_results['during & after_attack_window']['AUC_BDA']} \t"
                    results_bda_auc[schedule][attack][i, j] = auc_results['during & after_attack_window']['AUC_BDA']
                    results_50_lifespan[schedule][attack][i, j] = l_50

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

                if attack in ['A3FL', 'IBA']:
                    beta = 1
                elif attack == "Chameleon":
                    beta = 0.008 # This is the average beta, since the benign learning rate changes during the attack window
                elif attack == "FCBA":
                    beta = 0.5
                elif attack == "DarkFed":
                    beta = 0.05
                original_vals = {
                    #'mal_value': config['adversarial_optimizer'].get('lr_factor', config['adversarial_optimizer'].get('local_lr', config['optimizer']['local_lr'] / config['optimizer']['local_lr'])),
                    'mal_value': beta,
                    'bda_during': round(100 * temp['during_attack_window']['BDA'], 2),
                    'mta_during': round(100 * temp['during_attack_window']['MTA'], 2),
                    'bda_after': round(100 * temp['after_attack_window']['BDA'], 2),
                    'mta_after': round(100 * temp['after_attack_window']['MTA'], 2),
                    'bda_auc_during_after': auc_temp['during & after_attack_window']['AUC_BDA'],
                    '50_lifespan': int(l_50),
                }
            else:
                original_vals = None

            plot_graphs(malicious, benign, results_bda_during[schedule][attack], results_bda_after[schedule][attack],
                        results_bda_auc[schedule][attack], results_mta_during[schedule][attack],
                        results_mta_after[schedule][attack], results_50_lifespan[schedule][attack], r'\beta', r'\eta_b',
                        f'LR - {attack} ({schedule})')

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
        if schedule == "decay":
            for attack in attacks:
                attack_snake_case = "".join(["_" + c.lower() if c.isupper() else c.lower() for c in attack])
                if attack_snake_case[0] == "_":
                    attack_snake_case = attack_snake_case[1:]
                best_choices[attack] = {}
                for ben in benign:
                    best_index = np.argmax(results_bda_during[schedule][attack][:, indexOf(benign, ben)])
                    best_choices[attack][ben] = malicious[best_index]

        with open('./evaluation/hyper_params/results/best_mal_choices_greedy_lr.json', 'w') as f:
            json.dump(best_choices, f)

        balancing_param = 0.5
        best_choices = {}
        if schedule == "decay":
            for attack in attacks:
                attack_snake_case = "".join(["_" + c.lower() if c.isupper() else c.lower() for c in attack])
                if attack_snake_case[0] == "_":
                    attack_snake_case = attack_snake_case[1:]
                best_choices[attack] = {}
                for ben in benign:
                    total_acc = (1 - balancing_param) * results_bda_during[schedule][attack][:, indexOf(benign, ben)] + balancing_param * results_mta_during[schedule][attack][:, indexOf(benign, ben)]
                    best_index = np.argmax(total_acc)
                    best_choices[attack][ben] = malicious[best_index]

        with open('./evaluation/hyper_params/results/best_mal_choices_balanced_lr.json', 'w') as f:
            json.dump(best_choices, f)

        allowed_mta_decreases = [0.02, 0.05]
        for allowed_mta_decrease in allowed_mta_decreases:
            best_choices = {}
            if schedule == "decay":
                for attack in attacks:
                    attack_snake_case = "".join(["_" + c.lower() if c.isupper() else c.lower() for c in attack])
                    if attack_snake_case[0] == "_":
                        attack_snake_case = attack_snake_case[1:]
                    best_choices[attack] = {}
                    for ben in benign:
                        mtas = results_mta_during[schedule][attack][:, indexOf(benign, ben)]
                        bdas = results_bda_during[schedule][attack][:, indexOf(benign, ben)]
                        max_mta = mtas.max()
                        mta_constraint = max_mta - allowed_mta_decrease

                        # Get mask of indices that satisfy the MTA constraint
                        valid_indices = np.nonzero(mtas >= mta_constraint)[0]
                        # Among valid indices, select the one with the highest BDA
                        selected_index = valid_indices[np.argmax(bdas[valid_indices])]

                        best_choices[attack][ben] = malicious[selected_index]

            with open(f'./evaluation/hyper_params/results/best_mal_choices_MTA_constrained_{allowed_mta_decrease}_lr.json', 'w') as f:
                json.dump(best_choices, f)

    # Plot ablation of FCBA on MobileNetV2

    results_bda_during = np.zeros((len(malicious), len(benign)))
    results_bda_after = np.zeros((len(malicious), len(benign)))
    results_mta_during = np.zeros((len(malicious), len(benign)))
    results_mta_after = np.zeros((len(malicious), len(benign)))
    results_bda_auc = np.zeros((len(malicious), len(benign)))
    results_50_lifespan = np.zeros((len(malicious), len(benign)))

    for i, mal in enumerate(malicious):
        for j, ben in enumerate(benign):
            dir = f'logs/hyper_params/ablations/mobilenet_v2/FCBA_{ben}_{mal}'

            results = extract_accuracies(dir)

            results_bda_during[i, j] = results['during_attack_window']['BDA']
            results_bda_after[i, j] = results['after_attack_window']['BDA']
            results_mta_during[i, j] = results['during_attack_window']['MTA']
            results_mta_after[i, j] = results['after_attack_window']['MTA']

            l_50, l_m, l_wm, auc_results = extract_lifespans(dir)
            results_bda_auc[i, j] = auc_results['during & after_attack_window']['AUC_BDA']
            results_50_lifespan[i, j] = l_50

    plot_graphs(malicious, benign, results_bda_during, results_bda_after, results_bda_auc, results_mta_during, results_mta_after, results_50_lifespan, r'\beta', r'\eta_b', f'LR - FCBA MobileNetV2')

    # Plot ablation of FCBA on VGG17

    results_bda_during = np.zeros((len(malicious), len(benign)))
    results_bda_after = np.zeros((len(malicious), len(benign)))
    results_mta_during = np.zeros((len(malicious), len(benign)))
    results_mta_after = np.zeros((len(malicious), len(benign)))
    results_bda_auc = np.zeros((len(malicious), len(benign)))
    results_50_lifespan = np.zeros((len(malicious), len(benign)))

    for i, mal in enumerate(malicious):
        for j, ben in enumerate(benign):
            dir = f'logs/hyper_params/ablations/vgg17/FCBA_{ben}_{mal}'

            results = extract_accuracies(dir)

            results_bda_during[i, j] = results['during_attack_window']['BDA']
            results_bda_after[i, j] = results['after_attack_window']['BDA']
            results_mta_during[i, j] = results['during_attack_window']['MTA']
            results_mta_after[i, j] = results['after_attack_window']['MTA']

            l_50, l_m, l_wm, auc_results = extract_lifespans(dir)
            results_bda_auc[i, j] = auc_results['during & after_attack_window']['AUC_BDA']
            results_50_lifespan[i, j] = l_50

    plot_graphs(malicious, benign, results_bda_during, results_bda_after, results_bda_auc, results_mta_during, results_mta_after, results_50_lifespan, r'\beta', r'\eta_b', f'LR - FCBA VGG17')

    # Plot ablation of FCBA on Tiny-ImageNet

    results_bda_during = np.zeros((len(malicious), len(benign)))
    results_bda_after = np.zeros((len(malicious), len(benign)))
    results_mta_during = np.zeros((len(malicious), len(benign)))
    results_mta_after = np.zeros((len(malicious), len(benign)))
    results_bda_auc = np.zeros((len(malicious), len(benign)))
    results_50_lifespan = np.zeros((len(malicious), len(benign)))

    for i, mal in enumerate(malicious):
        for j, ben in enumerate(benign):
            dir = f'logs/hyper_params/ablations/imagenet/FCBA_{ben}_{mal}'

            results = extract_accuracies(dir)

            results_bda_during[i, j] = results['during_attack_window']['BDA']
            results_bda_after[i, j] = results['after_attack_window']['BDA']
            results_mta_during[i, j] = results['during_attack_window']['MTA']
            results_mta_after[i, j] = results['after_attack_window']['MTA']

            l_50, l_m, l_wm, auc_results = extract_lifespans(dir)
            results_bda_auc[i, j] = auc_results['during & after_attack_window']['AUC_BDA']
            results_50_lifespan[i, j] = l_50

    plot_graphs(malicious, benign, results_bda_during, results_bda_after, results_bda_auc, results_mta_during, results_mta_after, results_50_lifespan, r'\beta', r'\eta_b', 'LR - FCBA ImageNet')
