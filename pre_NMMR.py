from src.models.NMMR_trainers import NMMR_Trainer_Experiment_q
from src.models.NMMR_experiments import NMMR_experiment_h, NMMR_experiment_q
from src.data_type import grid_search_dict
import numpy as np
import pandas as pd
import os
import json
from pathlib import Path
from data.DGP import DGP, DGP_test
from src.data_type.data_class import NMMRTrainDataSet_h, NMMRTrainDataSetTorch_h, NMMRTestDataSetTorch_h, NMMRTrainDataSet_q, NMMRTrainDataSetTorch_q, NMMRTestDataSetTorch_q

os.chdir("C:/Users/shent/Desktop/Optimal-Individulized-Decison-making-with-Proxies2/")
data_dir = Path.cwd().joinpath('data')

config_path = Path.cwd().joinpath('configs/nmmr_selection_h.json')

with open(config_path) as f:
        config = json.load(f)

data_config = config["data"]
model_config = config["model"]

######Sample script used in NMMR for selecting penalty term########
##########partially adopted from https://github.com/beamlab-hsph/Neural-Moment-Matching-Regression/blob/main/src/experiment.py #######

X, Z, W, U, A, Y= DGP(900,1)
X_v, Z_v, W_v, U_v, A_v, Y_v = DGP(100,1)
train_data = NMMRTrainDataSet_h(treatment=A, treatment_proxy=Z,outcome_proxy=W,outcome=Y,backdoor=X)
train_torch = NMMRTrainDataSetTorch_h.from_numpy(train_data)
val_data = NMMRTrainDataSet_h(treatment=A_v, treatment_proxy=Z_v,outcome_proxy=W_v,outcome=Y_v,backdoor=X_v)
val_torch = NMMRTrainDataSetTorch_h.from_numpy(val_data)

X_t, Z_t, W_t, U_t = DGP_test(100)
test_data = NMMRTestDataSetTorch_h(treatment=np.array([[-1], [1]]),
                          outcome_proxy=W_t,
                          backdoor=X_t)
test_torch = NMMRTestDataSetTorch_h.from_numpy(test_data)

for dump_name, env_param in grid_search_dict(data_config):
        one_dump_dir = data_dir.joinpath(dump_name)
        os.mkdir(one_dump_dir)
        for mdl_dump_name, mdl_param in grid_search_dict(model_config):
            if mdl_dump_name != "one":
                one_mdl_dump_dir = one_dump_dir.joinpath(mdl_dump_name[-1])
                os.mkdir(one_mdl_dump_dir)
            else:
                one_mdl_dump_dir = one_dump_dir

            if model_config.get("log_metrics", False) == "True":
                test_losses = []
                train_metrics_ls = []
                for idx in range(config['n_repeat']):
                    test_loss, train_metrics = NMMR_experiment_h(train_torch, val_torch, test_torch, mdl_param, one_mdl_dump_dir, idx, 0)
                    train_metrics['rep_ID'] = idx
                    train_metrics_ls.append(train_metrics)
                    if test_loss is not None:
                        test_losses.append(test_loss)

                if test_losses:
                    np.savetxt(one_mdl_dump_dir.joinpath("result.csv"), np.array(test_losses))
                metrics_df = pd.concat(train_metrics_ls).reset_index()
                metrics_df.rename(columns={'index': 'epoch_num'}, inplace=True)
                metrics_df.to_csv(one_mdl_dump_dir.joinpath("train_metrics.csv"), index=False)