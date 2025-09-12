import re

from matplotlib import pyplot as plt

linestyles = ['-', '--', '-.', ':']

def plot_graphs(malicious, benign, results_bda_during, results_bda_after, results_bda_auc, results_mta_during,
                results_mta_after, results_50_lifespan, xlabel, trace_label, title, original_vals=None):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 16
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'

    ### Plot BDA during atk. window
    fig, axs = plt.subplots()
    for j, ben in enumerate(benign):
        axs.plot(malicious, 100 * results_bda_during[:, j], label=rf'${trace_label}={ben}$', linestyle=linestyles[j])

    if original_vals is not None:
        axs.scatter([original_vals['mal_value']], [original_vals['bda_during']], marker='x', label=rf'${trace_label}={original_vals["ben_value"]}$ (orig.)' if "ben_value" in original_vals else rf'original', zorder=len(benign) + 1, color='black')

    axs.set_ylim((0.0, 100.0))
    axs.set_ylabel(r'$\mathtt{BDA}$ [\%]')
    axs.set_xlabel(rf'${xlabel}$')
    axs.legend()
    axs.grid()
    plt.savefig(f'evaluation/hyper_params/results/{title} BDA During.pdf', bbox_inches='tight')
    plt.show()

    ### Plot BDA after atk. window
    fig, axs = plt.subplots()
    for j, ben in enumerate(benign):
        axs.plot(malicious, 100 * results_bda_after[:, j], label=rf'${trace_label}={ben}$', linestyle=linestyles[j])

    if original_vals is not None:
        axs.scatter([original_vals['mal_value']], [original_vals['bda_after']], marker='x', label=rf'${trace_label}={original_vals["ben_value"]}$ (orig.)' if "ben_value" in original_vals else rf'original', zorder=len(benign) + 1, color='black')

    axs.set_ylim((0.0, 100.0))
    axs.set_ylabel(r'$\mathtt{BDA^*}$ [\%]')
    axs.set_xlabel(rf'${xlabel}$')
    axs.legend()
    axs.grid()
    plt.savefig(f'evaluation/hyper_params/results/{title} BDA After.pdf', bbox_inches='tight')
    plt.show()

    ### Plot BDA during & after atk. window in one plot
    fig, axs = plt.subplots()
    for j, ben in enumerate(benign):
        line, = axs.plot(malicious, 100 * results_bda_during[:, j], label=rf'${trace_label}={ben}$', linestyle='-')
        axs.plot(malicious, 100 * results_bda_after[:, j], label='_nolegend_', linestyle='--', color=line.get_color())

    axs.set_ylim((0.0, 100.0))
    axs.set_ylabel(r'BDA [\%]')
    axs.set_xlabel(rf'${xlabel}$')
    axs.legend()
    axs.grid()
    plt.savefig(f'evaluation/hyper_params/results/{title} BDA.pdf', bbox_inches='tight')
    plt.show()

    ### Plot MTA during atk. window
    fig, axs = plt.subplots()
    for j, ben in enumerate(benign):
        axs.plot(malicious, 100 * results_mta_during[:, j], label=rf'${trace_label}={ben}$', linestyle=linestyles[j])

    if original_vals is not None:
        axs.scatter([original_vals['mal_value']], [original_vals['mta_during']], marker='x', label=rf'${trace_label}={original_vals["ben_value"]}$ (orig.)' if "ben_value" in original_vals else rf'original', zorder=len(benign) + 1, color='black')

    axs.set_ylim((0.0, 100.0))
    axs.set_ylabel(r'$\mathtt{MTA}$ [\%]')
    axs.set_xlabel(rf'${xlabel}$')
    axs.legend()
    axs.grid()
    plt.savefig(f'evaluation/hyper_params/results/{title} MTA During.pdf', bbox_inches='tight')
    plt.show()

    ### Plot MTA after atk. window
    fig, axs = plt.subplots()
    for j, ben in enumerate(benign):
        axs.plot(malicious, 100 * results_mta_after[:, j], label=rf'${trace_label}={ben}$', linestyle=linestyles[j])

    if original_vals is not None:
        axs.scatter([original_vals['mal_value']], [original_vals['mta_after']], marker='x', label=rf'${trace_label}={original_vals["ben_value"]}$ (orig.)' if "ben_value" in original_vals else rf'original', zorder=len(benign) + 1, color='black')

    axs.set_ylim((0.0, 100.0))
    axs.set_ylabel(r'$\mathtt{MTA^*}$ [\%]')
    axs.set_xlabel(rf'${xlabel}$')
    axs.legend()
    axs.grid()
    plt.savefig(f'evaluation/hyper_params/results/{title} MTA After.pdf', bbox_inches='tight')
    plt.show()

    ### Plot MTA during & after atk. window in one plot
    fig, axs = plt.subplots()
    for j, ben in enumerate(benign):
        line, = axs.plot(malicious, 100 * results_mta_during[:, j], label=rf'${trace_label}={ben}$', linestyle='-')
        axs.plot(malicious, 100 * results_mta_after[:, j], label='_nolegend_', linestyle='--', color=line.get_color())

    axs.set_ylim((0.0, 100.0))
    axs.set_ylabel(r'MTA [\%]')
    axs.set_xlabel(rf'${xlabel}$')
    axs.legend()
    axs.grid()
    plt.savefig(f'evaluation/hyper_params/results/{title} MTA.pdf', bbox_inches='tight')
    plt.show()

    ### Plot AuC_BDA during and after the atk. window
    fig, axs = plt.subplots()
    for j, ben in enumerate(benign):
        axs.plot(malicious, results_bda_auc[:, j], label=rf'${trace_label}={ben}$', linestyle=linestyles[j])

    if original_vals is not None:
        axs.scatter([original_vals['mal_value']], [original_vals['bda_auc_during_after']], marker='x', label=rf'${trace_label}={original_vals["ben_value"]}$ (orig.)' if "ben_value" in original_vals else rf'original', zorder=len(benign) + 1, color='black')

    axs.set_ylabel(r'$\mathtt{AuC}$')
    axs.set_xlabel(rf'${xlabel}$')
    axs.legend()
    axs.grid()
    fig.savefig(f'evaluation/hyper_params/results/{title} AuC.pdf', bbox_inches='tight')
    plt.show()

    ### Plot 50%-Lifespan
    fig, axs = plt.subplots()
    for j, ben in enumerate(benign):
        axs.plot(malicious, results_50_lifespan[:, j], label=rf'${trace_label}={ben}$', linestyle=linestyles[j])

    if original_vals is not None:
        axs.scatter([original_vals['mal_value']], [original_vals['50_lifespan']], marker='x', label=rf'${trace_label}={original_vals["ben_value"]}$ (orig.)' if "ben_value" in original_vals else rf'original', zorder=len(benign) + 1, color='black')

    axs.set_ylabel(r'$Span_{50}$')
    axs.set_xlabel(rf'${xlabel}$')
    axs.legend()
    axs.grid()
    plt.savefig(f'evaluation/hyper_params/results/{title} Lifespan.pdf', bbox_inches='tight')
    plt.show()


def extract_accuracies(path):
    with open(f'{path}/average_accs.log', 'r') as f:
        # Regex patterns to extract MTA and BDA during and after the attack window
        pattern = r"Accs (before|during|after) attack window \(MTA / BDA / BACC\): ([\w\.\-\d]+) / ([\w\.\-\d]+) / ([\w\.\-\d]+)"
        matches = re.findall(pattern, "".join(f.readlines()))

        # Dictionary to store the extracted values
        results = {}
        for match in matches:
            window, mta, bda, bacc = match
            results[f"{window}_attack_window"] = {"MTA": float(mta), "BDA": float(bda), "BACC": float(bacc)}

    return results


def extract_lifespans(path):
    with open(f'{path}/lifespans.log', 'r') as f:
        content = "".join(f.readlines())

        pattern = r"50 %:\s*(\d+)\.\d+"
        matches = re.findall(pattern, content)

        lifespan_50 = f'{matches[0]} \t'

        pattern = r"(Mean|Weighted Mean) Lifespan \[15%, 100%\]: (\d+\.\d+)"
        matches = re.findall(pattern, content)
        for match in matches:
            type, lifespan = match
            if type == "Mean":
                lifespan_mean = f'{round(float(lifespan), 2)} \t'
            elif type == "Weighted Mean":
                lifespan_weighted_mean = f'{round(float(lifespan), 2)} \t'

        pattern = r"AuC (during|after|during & after) attack window \[10%, 100%\] \(MTA / BDA\): ([^\s]*) / ([^\s]*)"
        matches = re.findall(pattern, content)
        auc_results = {}
        for match in matches:
            window, auc_mta, auc_bda = match
            auc_results[f'{window}_attack_window'] = {"AUC_MTA": float(auc_mta), "AUC_BDA": float(auc_bda)}

    return lifespan_50, lifespan_mean, lifespan_weighted_mean, auc_results
