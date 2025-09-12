import itertools

import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import yaml

from evaluation.hyper_params.utils import extract_accuracies, extract_lifespans

attacks = ['A3FL', 'Chameleon', 'DarkFed', 'FCBA']

experiments = []
for attack in attacks:
    experiments += [
            f'hyper_params/learning_rate/{attack}_{lr_b}_{beta}_decay' for (lr_b, beta) in itertools.product([0.05, 0.1, 0.2, 0.5], [0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
        ] + [
            f'hyper_params/local_epochs/{attack}_{E_b}_{E_m}' for (E_b, E_m) in itertools.product([2, 5, 10, 20], [2, 5, 10, 20])
        ] + [
            f'hyper_params/batch_size/{attack}_{B_b}_{B_m}' for (B_b, B_m) in itertools.product([32, 64, 128], [32, 64, 128])
        ] + [
            f'hyper_params/momentum/{attack}_{mu_b}_{mu_m}' for (mu_b, mu_m) in itertools.product([0.8, 0.9, 0.99], [0.8, 0.9, 0.99])
        ] + [
            f'hyper_params/weight_decay/{attack}_{wd_b}_{wd_m}' for (wd_b, wd_m) in itertools.product([0.0001, 0.0005, 0.001], [0.0001, 0.0005, 0.001])
        ]

def summary_to_latex(results, label="Model", rotate_label=False):
    lines = []

    # Handle rotated label or not
    if rotate_label:
        label_header = f"\\multirow{{{len(results.params)}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{label}}}}}"
    else:
        label_header = f"\\multirow{{{len(results.params)}}}{{*}}{{{label}}}"

    lines.append(label_header)

    for i, param in enumerate(results.params.index):
        coef = results.params[param]
        std_err = results.bse[param]
        t_val = results.tvalues[param]
        p_val = results.pvalues[param]
        conf_low, conf_high = results.conf_int().loc[param]

        p_str = f"{p_val:.3f}"
        if p_val < 0.001:
            p_str = "\\textbf{< 0.001}"
        elif p_val < 0.05:
            p_str = f"\\textbf{{{p_str}}}"

        param_str = {
            'const': '$\mathtt{const}$',
            'lr_b': '$\eta_b$',
            'mu_b': '$\mu_b$',
            'wd_b': '$\lambda_b$',
            'E_b': '$E_b$',
            'B_b': '$B_b$',
            'N': '$N$',
            'M': '$M$'
        }

        row = f"& {param_str[param]} & {coef:.4f} & {std_err:.3f} & {t_val:.3f} & {p_str} & {conf_low:.3f} & {conf_high:.3f} \\\\"
        if i != 0:
            row = "    " + row  # indent following rows
        lines.append(row)

    return "\n".join(lines)


if __name__ == '__main__':
    results = {
        'lr_b': [],
        'lr_m': [],
        'mu_b': [],
        'mu_m': [],
        'wd_b': [],
        'wd_m': [],
        'B_b': [],
        'B_m': [],
        'E_b': [],
        'E_m': [],
        'bda_during': [],
        'bda_after': [],
        'mta_during': [],
        'mta_after': [],
        'auc': [],
        'lifespan_50': [],
    }

    for experiment in experiments:
        dir = f'logs/{experiment}'
        with open(f'{dir}/config.yaml', "r") as file:
            configuration = yaml.safe_load(file)

        results['lr_b'].append(configuration.get('optimizer').get('local_lr'))
        if 'local_lr' in configuration['adversarial_optimizer']:
            results['lr_m'].append(configuration.get('adversarial_optimizer').get('local_lr'))
        elif 'lr_factor' in configuration['adversarial_optimizer']:
            results['lr_m'].append(configuration.get('adversarial_optimizer').get('lr_factor') * results['lr_b'][-1])
        else:
            results['lr_m'].append(results['lr_b'][-1])

        results['mu_b'].append(configuration.get('optimizer').get('momentum'))
        results['mu_m'].append(configuration.get('adversarial_optimizer').get('momentum'))

        results['wd_b'].append(configuration.get('optimizer').get('weight_decay'))
        results['wd_m'].append(configuration.get('adversarial_optimizer').get('weight_decay'))

        results['B_b'].append(configuration.get('batch_size'))
        results['B_m'].append(configuration.get('attacker').get('batch_size', results['B_b'][-1]))

        results['E_b'].append(configuration.get('local_epochs'))
        results['E_m'].append(configuration.get('local_epochs_malicious_clients'))

        res = extract_accuracies(dir)
        results['bda_during'].append(res['during_attack_window']['BDA'])
        results['bda_after'].append(res['after_attack_window']['BDA'])
        results['mta_during'].append(res['during_attack_window']['MTA'])
        results['mta_after'].append(res['after_attack_window']['MTA'])

        lifespan_50, _, _, auc_results = extract_lifespans(dir)
        results['auc'].append(float(auc_results['during & after_attack_window']['AUC_BDA']))
        results['lifespan_50'].append(float(lifespan_50))

    data = pd.DataFrame.from_dict(results)

    X_columns = ['lr_b', 'mu_b', 'wd_b', 'B_b', 'E_b']
    y_columns = ['mta_during', 'mta_after', 'bda_during', 'bda_after', 'auc', 'lifespan_50']

    X = data[X_columns]

    fig, axs = plt.subplots(len(X_columns), len(y_columns), figsize=(len(y_columns)*4, len(X_columns)*4))

    for i in range(len(X_columns)):
        for j in range(len(y_columns)):
            axs[i, j].scatter(data[X_columns[i]], data[y_columns[j]])
            axs[i, j].set_xlabel(X_columns[i])
            axs[i, j].set_ylabel(y_columns[j])

    plt.tight_layout()
    plt.show()

    print('='* 10 + " Cross-Correlation " + "=" * 10)

    # Compute correlation matrix
    corr_matrix = X.corr()

    correlation_threshold = 0.8

    # Find highly correlated pairs (|corr| > correlation_threshold, excluding self-correlation)
    high_corr_pairs = [
        (col1, col2, corr_matrix.loc[col1, col2])
        for col1 in corr_matrix.columns
        for col2 in corr_matrix.columns
        if col1 != col2 and abs(corr_matrix.loc[col1, col2]) > correlation_threshold
    ]

    # Remove duplicate pairs (A, B) and (B, A)
    seen = set()
    filtered_pairs = []
    for col1, col2, corr in high_corr_pairs:
        if (col2, col1) not in seen:
            filtered_pairs.append((col1, col2, corr))
            seen.add((col1, col2))

    # Print results
    if len(filtered_pairs) > 0:
        for col1, col2, corr in filtered_pairs:
            print(f"{col1} - {col2}: {corr:.2f}")
    print('No high correlation detected.')

    print('='* 10 + " Variance analysis " + "=" * 10)
    print(X.var().sort_values())

    # Analyze multicollinearity using VIF analysis
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    print('='* 10 + " VIF analysis " + "=" * 10)
    print(vif_data)

    std_devs = X.std()

    # Normalize to ensure numerical stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    X_scaled_df = sm.add_constant(X_scaled_df)

    for y_key in y_columns:

        print('='* 10 + f" Results for {y_key} " + "=" * 10)

        y = data[y_key]
        model = sm.OLS(y, X_scaled_df).fit()

        print(model.summary())

        print(summary_to_latex(model, y_key, True))

        # Set significance level
        significance_level = 0.05

        # Iterate over parameters and check significance
        for param, p_value in zip(model.params.index[1:], model.pvalues[1:]):
            if p_value < significance_level:
                coefficient = model.params[param]
                # Assuming you have a predefined change in the parameter (e.g., change x by y)
                change_in_x = 1  # Example: Change in x by 1 unit

                if 'E' in param:
                    absolute_change = 1
                elif 'B' in param:
                    absolute_change = 16
                elif 'lr' in param or 'mu' in param:
                    absolute_change = 0.1
                elif 'wd' in param:
                    absolute_change = 0.0005
                else:
                    absolute_change = std_devs[param]
                change_in_x = absolute_change / std_devs[param]

                change_in_y = coefficient * change_in_x  # Dependent variable change
                significance_level_str = f"significance level: {p_value:.4f}"

                # Print the result
                if y_key not in ['auc', 'lifespan_50']:
                    print(f"Changing parameter {param} by {change_in_x} stddev (= {round(change_in_x * std_devs[param], 4)}) changes the {y_key} by {change_in_y * 100:.2f} % ({significance_level_str})")
                else:
                    print(f"Changing parameter {param} by {change_in_x} stddev (= {round(change_in_x * std_devs[param], 4)}) changes the {y_key} by {change_in_y:.2f} ({significance_level_str})")

        # Check residuals
        residuals = model.resid
        fitted_values = model.fittedvalues
        plt.figure(figsize=(8, 5))
        plt.scatter(fitted_values, residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("Fitted values")
        plt.ylabel("Residuals")
        plt.title(f"Residuals vs. Fitted values for {y_key}")
        plt.show()
