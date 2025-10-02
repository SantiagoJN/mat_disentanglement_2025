##################### TO BE USED BEFORE THE GET_METRICS.PY #####################

import os
import subprocess

models_path = "/mnt/cephfs/home/graphics/sjimenez/disentangling-vae-master/results"

ok_models = 0
for model_folder in os.listdir(models_path):
    content = os.listdir(f'results/{model_folder}')
    # if "model.pt" in content and "intermediate" not in model_folder: # The current model folder contains the model.pt file -> has ended its execution
    if "model.pt" in content and "test91_" in model_folder and not "OK" in model_folder:
        print(f'Getting visualizations of model {model_folder}')
        
        os.system(f'python main_viz.py {model_folder} all -r 8 -c 7')

        old = f'results/{model_folder}'
        new = f'results/OK_{model_folder}'
        os.rename(old, new) # Change the name to identify successful models easily
        ok_models += 1
        print('Done!')

print(f'Computed visualizations for a total of {ok_models} models.')

