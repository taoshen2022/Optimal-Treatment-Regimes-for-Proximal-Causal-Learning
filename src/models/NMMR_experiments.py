import os.path as op
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch

from src.data_type.data_class import NMMRTrainDataSetTorch_h, NMMRTestDataSetTorch_h, NMMRTestDataSetTorch_h, NMMRTrainDataSetTorch_q, NMMRTestDataSetTorch_q, NMMRTestDataSetTorch_q
from src.models.NMMR_trainers import NMMR_Trainer_Experiment_h, NMMR_Trainer_Experiment_q


def NMMR_experiment_h(train_data, val_data, test_data, model_config: Dict[str, Any],
                    dump_dir: Path,
                    random_seed: int = 42, verbose: int = 0):

    ######bridge function estimation (for tuning parameters, a script is provided in pre_NMMR.py) #######
    ##########partially adopted from https://github.com/beamlab-hsph/Neural-Moment-Matching-Regression/blob/main/src/experiment.py #######


    #params:
    #train_data, val_data, test_data: data used for training, validation and testing
    #model_config: document for different parameter setup (see nmmr_selection.json as an example)
    #dump_dir: directory path for storing the training and validation results

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)


    # convert datasets to Torch (for GPU runtime)
    train_t = NMMRTrainDataSetTorch_h.from_numpy(train_data)
    val_data_t = NMMRTrainDataSetTorch_h.from_numpy(val_data)


    # retrieve the trainer for this experiment
    trainer = NMMR_Trainer_Experiment_h(model_config, random_seed, dump_dir)

    # train model
    model = trainer.train(train_data, val_data, verbose)

    # prepare test data on the gpu
    if trainer.gpu_flg:
        test_data_t = test_data.to_gpu()
        val_data_t = val_data.to_gpu()

    E_wx_hawx = trainer.predict(model, test_data_t)

    pred = E_wx_hawx.detach().numpy()
    np.save(dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)

    if hasattr(test_data, 'structural'):
        np.testing.assert_array_equal(pred.shape, test_data.structural.shape)
        oos_loss = np.mean((pred - test_data.structural) ** 2)
    else:
        oos_loss = None

    if trainer.log_metrics:
        return oos_loss, pd.DataFrame(
            data={'causal_loss_train': torch.Tensor(trainer.causal_train_losses[-50:], device="cpu").numpy(),
                  'causal_loss_val': torch.Tensor(trainer.causal_val_losses[-50:], device="cpu").numpy()})
    else:
        return oos_loss

def NMMR_experiment_q(train_data, val_data, test_data, model_config: Dict[str, Any],
                    one_mdl_dump_dir: Path,
                    random_seed: int = 42, verbose: int = 0):
    ######bridge function estimation (for tuning parameters, a script is provided in pre_NMMR.py) #######
    ##########partially adopted from https://github.com/beamlab-hsph/Neural-Moment-Matching-Regression/blob/main/src/experiment.py #######


    #params:
    #train_data, val_data, test_data: data used for training, validation and testing
    #model_config: document for different parameter setup (see nmmr_selection.json as an example)
    #dump_dir: directory path for storing the training and validation results

    # set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)


    # convert datasets to Torch (for GPU runtime)
    train_t = NMMRTrainDataSetTorch_q.from_numpy(train_data)
    val_data_t = NMMRTrainDataSetTorch_q.from_numpy(val_data)


    # retrieve the trainer for this experiment
    trainer = NMMR_Trainer_Experiment_q(model_config, random_seed, one_mdl_dump_dir)

    # train model
    model = trainer.train(train_data, val_data, verbose)

    # prepare test data on the gpu
    if trainer.gpu_flg:
        test_data_t = test_data.to_gpu()
        val_data_t = val_data.to_gpu()

    E_zx_qazx = trainer.predict(model, test_data_t)

    pred = E_zx_qazx.detach().numpy()
    np.save(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)

    if hasattr(test_data, 'structural'):
        np.testing.assert_array_equal(pred.shape, test_data.structural.shape)
        oos_loss = np.mean((pred - test_data.structural) ** 2)
    else:
        oos_loss = None

    if trainer.log_metrics:
        return oos_loss, pd.DataFrame(
            data={'causal_loss_train': torch.Tensor(trainer.causal_train_losses[-50:], device="cpu").numpy(),
                  'causal_loss_val': torch.Tensor(trainer.causal_val_losses[-50:], device="cpu").numpy()})
    else:
        return oos_loss
