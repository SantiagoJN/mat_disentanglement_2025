##################### TO BE USED AFTER THE GET_VISUALIZATIONS.PY #####################

import os
import subprocess

models_path = "/mnt/cephfs/home/graphics/sjimenez/disentangling-vae-master/results"

ok_models = 0
for model_folder in os.listdir(models_path):
    if "OK" in model_folder: # Models that completed the training
        print(f'Getting metrics of model {model_folder}')
        
        #*** METRIC USING TEST DATASET ***
        interpreter = "/mnt/cephfs/home/graphics/sjimenez/anaconda3/envs/env_metrics/bin/python3.7"
        script = "/mnt/cephfs/home/graphics/sjimenez/lib_disentanglement/disentanglement_lib/evaluation/custom_factor_vae_test.py"
        dataset = "masked-hpo-test" # ! Using the TEST dataset !
        num_trials = "10"
        exp_name = f"{model_folder}"
        cmd = f"{interpreter} {script} {exp_name} {dataset} {num_trials}"
        output = subprocess.check_output(cmd, shell=True)

        output = output.decode("utf-8")

        factorVAE_idx = output.find("FactorVAE:")
        MIS_idx = output.find("Mutual Info Score:")
        fVAE = float(output[factorVAE_idx+11:factorVAE_idx+16])
        MIS = float(output[MIS_idx+19:MIS_idx+24])

        metric_test = fVAE + (1.0 - MIS)
        print(f'Metric: {metric_test}')

        #*** METRIC USING VALIDATION DATASET ***
        interpreter = "/mnt/cephfs/home/graphics/sjimenez/anaconda3/envs/env_metrics/bin/python3.7"
        script = "/mnt/cephfs/home/graphics/sjimenez/lib_disentanglement/disentanglement_lib/evaluation/custom_factor_vae_test.py"
        dataset = "masked-validation" # ! Using the VALIDATION dataset !
        num_trials = "10"
        exp_name = f"{model_folder}"
        cmd = f"{interpreter} {script} {exp_name} {dataset} {num_trials}"
        output = subprocess.check_output(cmd, shell=True)

        output = output.decode("utf-8")

        factorVAE_idx = output.find("FactorVAE:")
        MIS_idx = output.find("Mutual Info Score:")
        fVAE = float(output[factorVAE_idx+11:factorVAE_idx+16])
        MIS = float(output[MIS_idx+19:MIS_idx+24])

        metric_val = fVAE + (1.0 - MIS)
        print(f'Metric: {metric_val}')


        old = f'results/{model_folder}'
        new = f'results/OK_val-{metric_val:.5f}_test-{metric_test:.5f}_{model_folder[3:]}'
        os.rename(old, new) # Change the name to identify successful models easily
        ok_models += 1
        print('Done!')

print(f'Computed visualizations for a total of {ok_models} models.')
