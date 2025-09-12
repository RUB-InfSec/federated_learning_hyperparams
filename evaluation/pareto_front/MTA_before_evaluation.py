import os

from evaluation.hyper_params.utils import extract_accuracies

if __name__ == "__main__":
    pareto_path = './logs/hyper_params/pareto'
    attacks = ['A3FL', 'Chameleon', 'DarkFed', 'FCBA']
    defenses = ['Bulyan', 'FoolsGold', 'Krum', 'MultiKrum', 'None']
    recommended_params = (0.15, 0.0005, 10, 32)

    for defense in defenses:
        mtas = []
        for attack in attacks:
            path = os.path.join(pareto_path, f'{attack}_{defense}_{recommended_params[0]}_{recommended_params[1]}_{recommended_params[2]}_{recommended_params[3]}')
            if os.path.exists(os.path.join(path, 'average_accs.log')):
                results = extract_accuracies(path)
                mtas.append(results["before_attack_window"]["MTA"])
                print(f'For {attack}/{defense}, we MTA_before={round(100*results["before_attack_window"]["MTA"], 2)} %')
        print(f'=> For {defense}, the average MTA_before is {round(100*sum(mtas)/len(mtas), 2)} %')
        print()