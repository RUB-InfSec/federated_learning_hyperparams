import json
import math
import os
import yaml
import pandas as pd
import numpy as np

from evaluation.hyper_params.utils import extract_accuracies, extract_lifespans

attacks_to_casing = {
    'a3fl': 'A3FL',
    'chameleon': 'Chameleon',
    'darkfed': 'DarkFed',
    'fcba': 'FCBA'
}

defenses_to_casing = {
    'bulyan': 'Bulyan',
    'foolsgold': 'FoolsGold',
    'krum': 'Krum',
}

def get_config(path):
    with open(os.path.join(path, 'config.yaml')) as f:
        return yaml.safe_load(f)

def get_attack_defense_from_config(config):
    attack = attacks_to_casing[config['attacker']['name']]
    if config['defense']['name'] == 'krum' and config['defense']['m'] > 0:
        defense = 'MultiKrum'
    elif config['defense']['name'] is None:
        defense = 'None'
    else:
        defense = defenses_to_casing.get(config['defense']['name'], None)
    return (attack, defense)


def load_data_for_simple_adversary(dir: str) -> pd.DataFrame:
    rows = []

    for experiment in os.listdir(dir):
        path = os.path.join(dir, experiment)
        if os.path.exists(os.path.join(path, 'average_accs.log')):
            results = extract_accuracies(path)
            config = get_config(path)
            attack, defense = get_attack_defense_from_config(config)
            lifespan_50, _, _, _ = extract_lifespans(path)

            row = {
                'Attack': attack,
                'Defense': defense,
                'lr_b': config['optimizer']['local_lr'],
                'mu_b': config['optimizer']['momentum'],
                'wd_b': config['optimizer']['weight_decay'],
                'E_b': config['local_epochs'],
                'B_b': config['batch_size'],
                'MTA_during': results['during_attack_window']['MTA'],
                'BDA_during': results['during_attack_window']['BDA'],
                'MTA_after': results['after_attack_window']['MTA'],
                'BDA_after': results['after_attack_window']['BDA'],
                'lifespan_50': int(lifespan_50),
            }

            rows.append(row)

    return pd.DataFrame(rows)

def load_data_for_NSGA_adversary(dir: str, min_mta: float = 0.0, max_gen_mal = math.inf) -> pd.DataFrame:
    rows = []

    print(f'Command lines for {dir=}, {min_mta=}, {max_gen_mal=} are:')

    for attack in attacks:
        for defense in defenses:
            subdir = os.path.join(dir, f'{attack}_{defense}_{optimal_params["eta_b"]}_{optimal_params["lambda_b"]}_{optimal_params["E_b"]}_{optimal_params["B_b"]}')

            if os.path.exists(subdir):

                best_accs = (0.0, 0.0)
                best_acc_path = ''

                if os.path.exists(os.path.join(dir, f'{attack}_{defense}', 'generations_to_configurations.json')):
                    with open(os.path.join(dir, f'{attack}_{defense}', 'generations_to_configurations.json'), 'r') as f:
                        generations_to_configurations = json.load(f)

                    for gen_mal, trials in generations_to_configurations.items():
                        if int(gen_mal) < max_gen_mal:
                            for trial in trials:
                                path = "/".join(trial.split('/')[-2:])
                                if os.path.exists(os.path.join(dir, path, 'average_accs.log')):
                                    results = extract_accuracies(os.path.join(dir, path))
                                    mta, bda = results['during_attack_window']['MTA'], results['during_attack_window']['BDA']
                                    if bda > best_accs[1] and mta >= min_mta:
                                        best_accs = (mta, bda)
                                        best_acc_path = os.path.join(dir, path)
                else:
                    if max_gen_mal < math.inf:
                        print(f'max_gen_mal is specified but the given subfolder ({os.path.join(dir, f"{attack}_{defense}", "generations_to_configurations.json")}) has no generation_to_configurations.json! Is this really what you wanted?')

                    for experiment in os.listdir(subdir):
                        path = os.path.join(subdir, experiment)
                        if os.path.exists(os.path.join(path, 'average_accs.log')):
                            results = extract_accuracies(path)
                            mta, bda = results['during_attack_window']['MTA'], results['during_attack_window']['BDA']
                            if bda > best_accs[1] and mta >= min_mta:
                                best_accs = (mta, bda)
                                best_acc_path = path

                if best_acc_path != '':
                    results = extract_accuracies(best_acc_path)
                    config = get_config(best_acc_path)
                    attack, defense = get_attack_defense_from_config(config)
                    lifespan_50, _, _, _ = extract_lifespans(best_acc_path)

                    print(f"python multi_run.py --base_config experiments/hyper_params/attacks_vs_defenses_pareto/{attack}_{defense}.yaml --lr_b {config['optimizer']['local_lr']} --mu_b {config['optimizer']['momentum']} --wd_b {config['optimizer']['weight_decay']}  --E_b {config['local_epochs']} --B_b {config['batch_size']} --beta {config['adversarial_optimizer']['lr_factor']} --mu_m {config['adversarial_optimizer']['momentum']} --wd_m {config['adversarial_optimizer']['weight_decay']} --E_m {config['local_epochs_malicious_clients']} --B_m {config['attacker']['batch_size']} --experiment_descriptor_template hyper_params/recommended/NSGA_unconstrained/{attack}_{defense}_{{lr_b}}_{{wd_b}}_{{E_b}}_{{B_b}}/{{beta}}_{{wd_m}}_{{E_m}}_{{B_m}} --devices 0 --tasks_per_gpu 1 --test_batch_size 128 --num_workers {'2' if attack != 'A3FL' else '0'} &")

                    row = {
                        'Attack': attack,
                        'Defense': defense,
                        'lr_b': config['optimizer']['local_lr'],
                        'mu_b': config['optimizer']['momentum'],
                        'wd_b': config['optimizer']['weight_decay'],
                        'E_b': config['local_epochs'],
                        'B_b': config['batch_size'],
                        'MTA_during': results['during_attack_window']['MTA'],
                        'BDA_during': results['during_attack_window']['BDA'],
                        'MTA_after': results['after_attack_window']['MTA'],
                        'BDA_after': results['after_attack_window']['BDA'],
                        'lifespan_50': int(lifespan_50),
                    }
                else:
                    row = {
                        'Attack': attack,
                        'Defense': defense,
                        'lr_b': optimal_params["eta_b"],
                        'mu_b': optimal_params["mu_b"],
                        'wd_b': optimal_params["lambda_b"],
                        'E_b': optimal_params["E_b"],
                        'B_b': optimal_params["B_b"],
                        'MTA_during': np.nan,
                        'BDA_during': np.nan,
                        'MTA_after': np.nan,
                        'BDA_after': np.nan,
                        'lifespan_50': np.nan,
                    }

                rows.append(row)

    print()

    return pd.DataFrame(rows)


def format_mixed(val, data_type="float"):
    na_rep = "N.A."

    if isinstance(val, tuple):
        if val[0] == val[1]:
            res = f"{val[0]:.0f}" if data_type == "int" else f"{val[0]:.1f} \\%"
        else:
            res = f"{val[0]:.0f} / {val[1]:.0f}" if data_type == "int" else f"{val[0]:.1f} \\% / {val[1]:.1f} \\%"
    else:
        res = f"{val:.0f}" if data_type == "int" else f"{val:.1f} \\%"
    return res.replace("nan", na_rep).replace(f"{na_rep} \\%", na_rep)


if __name__ == "__main__":
    optimal_params = {'eta_b': 0.15, 'mu_b': 0.9, 'lambda_b': 0.0005, 'E_b': 10, 'B_b': 32}

    min_mta = 0.805

    attacks = ['A3FL', 'Chameleon', 'DarkFed', 'FCBA']
    defenses = ['Bulyan', 'FoolsGold', 'Krum', 'MultiKrum', 'None']

    experiments_path_greedy = './logs/hyper_params/recommended/greedy'
    experiments_path_recommended_vs_MTA_constrained_0_05 = './logs/hyper_params/recommended/MTA_constrained_0.05'
    experiments_path_recommended_vs_orig = './logs/hyper_params/recommended_benign'
    experiments_path_orig_vs_orig = './logs/hyper_params/original_benign'
    experiments_path_NSGA_II_unconstrained_36_trials = './logs/hyper_params/NSGA-II_adversary/unconstrained'
    experiments_path_NSGA_II_unconstrained_36_trials_long = './logs/hyper_params/recommended/NSGA_unconstrained'

    df_recommended_vs_greedy = load_data_for_simple_adversary(experiments_path_greedy)
    df_recommended_vs_greedy['Benign param.'] = 'Recommended'
    df_recommended_vs_greedy['Malicious param.'] = 'Simple_greedy'

    df_recommended_vs_MTA_constrained_0_05 = load_data_for_simple_adversary(experiments_path_recommended_vs_MTA_constrained_0_05)
    df_recommended_vs_MTA_constrained_0_05['Benign param.'] = 'Recommended'
    df_recommended_vs_MTA_constrained_0_05['Malicious param.'] = 'Simple_MTA'

    # Remove all values from df_recommended_vs_MTA_constrained_0_05 that are below 80% MTA
    df_recommended_vs_MTA_constrained_0_05[df_recommended_vs_MTA_constrained_0_05['MTA_during'] < min_mta] = np.nan

    df_orig_vs_orig = load_data_for_simple_adversary(experiments_path_orig_vs_orig)
    df_orig_vs_orig['Benign param.'] = 'Orig.'
    df_orig_vs_orig['Malicious param.'] = 'Orig.'

    df_recommended_vs_NSGA_unconstrained_36_trials = load_data_for_NSGA_adversary(experiments_path_NSGA_II_unconstrained_36_trials, max_gen_mal=4)
    df_recommended_vs_NSGA_unconstrained_36_trials['Benign param.'] = 'Recommended'
    df_recommended_vs_NSGA_unconstrained_36_trials['Malicious param.'] = '$\\text{Global}_\\text{g,36}$'

    df_recommended_vs_NSGA_MTA_constrained_0_05_36_trials = load_data_for_NSGA_adversary(experiments_path_NSGA_II_unconstrained_36_trials, min_mta, max_gen_mal=4)
    df_recommended_vs_NSGA_MTA_constrained_0_05_36_trials['Benign param.'] = 'Recommended'
    df_recommended_vs_NSGA_MTA_constrained_0_05_36_trials['Malicious param.'] = '$\\text{Global}_\\text{M,36}$'

    df_recommended_vs_NSGA_unconstrained_36_trials_long = load_data_for_NSGA_adversary(experiments_path_NSGA_II_unconstrained_36_trials_long, max_gen_mal=4)
    df_recommended_vs_NSGA_unconstrained_36_trials_long['Benign param.'] = 'Recommended'
    df_recommended_vs_NSGA_unconstrained_36_trials_long['Malicious param.'] = '$\\text{Global}_\\text{g,36}$'

    df_recommended_vs_NSGA_MTA_constrained_0_05_36_trials_long = load_data_for_NSGA_adversary(experiments_path_NSGA_II_unconstrained_36_trials_long, min_mta, max_gen_mal=4)
    df_recommended_vs_NSGA_MTA_constrained_0_05_36_trials_long['Benign param.'] = 'Recommended'
    df_recommended_vs_NSGA_MTA_constrained_0_05_36_trials_long['Malicious param.'] = '$\\text{Global}_\\text{M,36}$'

    df_all = pd.concat([
        df_recommended_vs_greedy,
        df_recommended_vs_MTA_constrained_0_05,
        df_orig_vs_orig,
        #df_recommended_vs_NSGA_unconstrained_36_trials,
        #df_recommended_vs_NSGA_MTA_constrained_0_05_36_trials,
        df_recommended_vs_NSGA_unconstrained_36_trials_long,
        df_recommended_vs_NSGA_MTA_constrained_0_05_36_trials_long,
    ])

    pivot = df_all.pivot_table(
        index=['Attack', 'Defense'],
        columns=['Benign param.', 'Malicious param.'],
        values=['MTA_during', 'BDA_during', 'MTA_after', 'BDA_after', 'lifespan_50']
    )

    # Add Average row
    mean_row_full = pivot.mean(axis=0, numeric_only=True)
    mean_index_full = pd.MultiIndex.from_tuples([('Average', '')], names=pivot.index.names)
    mean_df_full = pd.DataFrame([mean_row_full.values], index=mean_index_full, columns=pivot.columns)

    # Add Average without defenses
    mean_row_with_defenses = pivot[pivot.index.get_level_values('Defense') == 'None'].mean(axis=0, numeric_only=True)
    mean_index_with_defenses = pd.MultiIndex.from_tuples([('Average w/o defenses', '')], names=pivot.index.names)
    mean_df_with_defenses = pd.DataFrame([mean_row_with_defenses.values], index=mean_index_with_defenses, columns=pivot.columns)

    # Add Average without defenses
    mean_row_wo_defenses = pivot[pivot.index.get_level_values('Defense') != 'None'].mean(axis=0, numeric_only=True)
    mean_index_wo_defenses = pd.MultiIndex.from_tuples([('Average w/ defenses', '')], names=pivot.index.names)
    mean_df_wo_defenses = pd.DataFrame([mean_row_wo_defenses.values], index=mean_index_wo_defenses, columns=pivot.columns)

    # Add rows to Pivot Table
    pivot = pd.concat([pivot, mean_df_full, mean_df_wo_defenses, mean_df_with_defenses])
    
    # Round floating point values
    cols_to_round = ['BDA_during', 'MTA_during', 'BDA_after', 'MTA_after']

    pivot.loc[:, cols_to_round] = (100 * pivot.loc[:, cols_to_round]).round(1)

    desired_order_first_level = ['MTA_during', 'BDA_during', 'MTA_after', 'BDA_after', 'lifespan_50']  # Put MTA first
    desired_order_second_level = ['Recommended', 'Orig.']  # Put recommended first
    desired_order_third_level = ['Simple', '\\nsga']

    # Merge columns
    for metric in desired_order_first_level:
        for benign_param in desired_order_second_level:
            if (col1 := (metric, benign_param, '$\\text{Global}_\\text{g,36}$')) in pivot.columns and (col2 := (metric, benign_param, '$\\text{Global}_\\text{M,36}$')) in pivot.columns:
                new_col = (metric, benign_param, '\\nsga')
                pivot[new_col] = list(zip(pivot[col1], pivot[col2]))
                pivot.drop(columns=[col1, col2], inplace=True)
            if (col1 := (metric, benign_param, 'NSGA_greedy')) in pivot.columns and (col2 := (metric, benign_param, 'NSGA_MTA')) in pivot.columns:
                new_col = (metric, benign_param, '\\nsga_{small}')
                pivot[new_col] = list(zip(pivot[col1], pivot[col2]))
                pivot.drop(columns=[col1, col2], inplace=True)
            if (col1 := (metric, benign_param, 'Simple_greedy')) in pivot.columns and (col2 := (metric, benign_param, 'Simple_MTA')) in pivot.columns:
                new_col = (metric, benign_param, 'Simple')
                pivot[new_col] = list(zip(pivot[col1], pivot[col2]))
                pivot.drop(columns=[col1, col2], inplace=True)

    # Build the new MultiIndex column order
    new_columns = []
    for super_col_1 in desired_order_first_level:
        for super_col_2 in desired_order_second_level:
            if super_col_2 == 'Recommended':
                for sub_col in desired_order_third_level:
                    new_columns.append((super_col_1, super_col_2, sub_col))
            else:
                for sub_col in pivot[super_col_1, super_col_2].columns:
                    new_columns.append((super_col_1, super_col_2, sub_col))

    # Reorder columns
    pivot = pivot.loc[:, pd.MultiIndex.from_tuples(new_columns)]

    # Join row index
    pivot.index = pivot.index.map(lambda x: f"{x[0]} / {x[1]}")
    pivot.index.name = "Attack_Defense"

    styler = pivot.style

    column_format = {
        col: lambda x: format_mixed(x, "float") for col in pivot.columns if col[0] != 'lifespan_50'
    }
    column_format.update({
        col: lambda x: format_mixed(x, "int") for col in pivot.columns if col[0] == 'lifespan_50'
    })
    styler.format(column_format, na_rep="N.A.")

    pivot.rename(columns={
        'MTA_during': '\\mta', 
        'BDA_during': '\\bda',
        'MTA_after': '\\mtaafter',
        'BDA_after': '\\bdaafter',
        'lifespan_50': '\\lifespan',
        'Simple_greedy': '$\\text{Global}_\\text{max}$',
        'Simple_MTA': '$\\text{Global}_\\text{c}$',
        'NSGA_greedy': '$\\nsga_\\text{max}$',
        'NSGA_MTA': '$\\nsga_\\text{c}$',
        'Simple': 'Greedy',
    }, inplace=True)
    pivot.columns.names = [None, '\\textbf{Benign param.}', '\\textbf{Malicious param.}']
    pivot.index.names = ['\\textbf{Attack/Defense}']

    latex_code = styler.to_latex(
        multicol_align='c|',
        hrules=True,
        column_format='|l|' + "cc|c|" * (len(pivot.columns) // 3)
    )

    latex_code = latex_code.replace('\\bottomrule', '\\hline')
    latex_code = latex_code.replace('\\midrule', '\\hline')
    latex_code = latex_code.replace('\\toprule', '\\hline')

    lines = latex_code.splitlines()

    lines[2] += f'\\cline{{2 - {(len(pivot.columns) // 3) * 3 + 1}}}'
    lines[3] += f'\\cline{{2 - {(len(pivot.columns) // 3) * 3 + 1}}}'

    print("\n".join(lines))
