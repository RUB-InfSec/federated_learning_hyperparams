import argparse
import json

import pandas

def smart_format(x):
    formatted = f"{x:.4f}".rstrip("0").rstrip(".")
    return f"{formatted}"


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--malicious_strategy', type=str, choices=['greedy', 'balanced', 'MTA_constrained_0.02', 'MTA_constrained_0.05'],  help="The strategy that is used to determine the malicious hyperparameters", default='greedy')
    args = parser.parse_args()

    parameters = {
        'lr': [0.05, 0.1, 0.2, 0.5],
        'E': [2, 5, 10, 20],
        'B': [32, 64, 128],
        'wd': [0.0001, 0.0005, 0.001],
    }

    parameter_to_latex = {
        'lr': '$\\eta$',
        'E': '$E$',
        'B': '$B$',
        'wd': '$\\lambda$',
    }

    attacks = ['A3FL', 'Chameleon', 'DarkFed', 'FCBA']

    rows = []

    for parameter in parameters:
        with open(f'evaluation/hyper_params/results/best_mal_choices_{args.malicious_strategy}_{parameter}.json', 'r') as f:
            best_vals = json.load(f)

        for val in parameters[parameter]:
            for attack in attacks:
                rows.append({
                    'parameter': parameter_to_latex[parameter],
                    'attack': attack,
                    'ben_value': val,
                    'mal_value': best_vals[attack][str(val)] * val if parameter == 'lr' else best_vals[attack][str(val)],
                })

    df = pandas.DataFrame(rows)

    pivot = df.pivot_table(
        index=['parameter', 'ben_value'],
        columns=['attack'],
        values=['mal_value'],
    )

    # Sort
    desired_order = ['$\\eta$', '$E$', '$B$', '$\\lambda$']
    pivot = pivot.sort_values(
        by=['parameter', 'ben_value'],
        key=lambda col: col.map({v: i for i, v in enumerate(desired_order)}) if col.name == 'parameter' else col
    )

    # Apply formatting to index level 'ben_value'
    pivot.index = pivot.index.set_levels([
        pivot.index.levels[0],  # 'parameter'
        pivot.index.levels[1].map(smart_format)  # 'ben_value'
    ])

    # Apply formatting to the DataFrame values
    pivot = pivot.map(smart_format)

    styler = pivot.style

    latex_code = styler.to_latex(
        hrules=True,
        column_format='cc|ccc'
    )

    latex_code = latex_code.replace('\\bottomrule', '\\hline')
    latex_code = latex_code.replace('\\midrule', '\\hline')
    latex_code = latex_code.replace('\\toprule', '\\hline')

    lines = latex_code.splitlines()

    for i, line in enumerate(lines):
        if "multirow" in line and "hline" not in lines[i-1]:
            lines[i-1] += "\\hline"

    print("\n".join(lines))
