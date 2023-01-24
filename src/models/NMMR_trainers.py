import os.path as op
from typing import Optional, Dict, Any
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from src.data_type.data_class import NMMRTrainDataSetTorch_h, NMMRTestDataSet_h, NMMRTrainDataSetTorch_q, NMMRTestDataSet_q
from src.models.NMMR_loss import NMMR_loss
from src.models.NMMR_model import MLP_for_NMMR
from src.models.kernel_utils import calculate_kernel_matrix

######Trainers used in NMMR########
##########partially adopted from https://github.com/beamlab-hsph/Neural-Moment-Matching-Regression/blob/main/src/models/NMMR/NMMR_trainers.py #####


class NMMR_Trainer_Experiment_h:
    def __init__(self, train_params: Dict[str, Any], random_seed: int, dump_folder: Optional[Path] = None):
        self.train_params = train_params
        self.n_epochs = train_params['n_epochs']
        self.batch_size = train_params['batch_size']
        self.gpu_flg = torch.cuda.is_available()
        self.log_metrics = train_params['log_metrics'] == "True"
        self.l2_penalty = train_params['l2_penalty']
        self.learning_rate = train_params['learning_rate']
        self.loss_name = train_params['loss_name']

        self.mse_loss = nn.MSELoss()

        if self.log_metrics:
            self.writer = SummaryWriter(log_dir=op.join(dump_folder, f"tensorboard_log_{random_seed}"))
            self.causal_train_losses = []
            self.causal_val_losses = []

    def compute_kernel(self, kernel_inputs):
        return calculate_kernel_matrix(kernel_inputs)
    
    def create(self, train_t: NMMRTrainDataSetTorch_h):
        input_size = 1 + 1 + train_t.backdoor.shape[1]
        model = MLP_for_NMMR(input_dim=input_size, train_params=self.train_params)
        return model

    def train(self, train_t: NMMRTrainDataSetTorch_h, val_t: NMMRTrainDataSetTorch_h, verbose: int = 0):
        # inputs consist of (A, W, X) tuples
        input_size = 1 + 1 + train_t.backdoor.shape[1]
        model = MLP_for_NMMR(input_dim=input_size, train_params=self.train_params)

        if self.gpu_flg:
            train_t = train_t.to_gpu()
            val_t = val_t.to_gpu()
            model.cuda()

        # weight_decay implements L2 penalty
        optimizer = optim.Adam(list(model.parameters()), lr=self.learning_rate, weight_decay=self.l2_penalty)

        n_sample = train_t.treatment.shape[0]
        # train model
        for epoch in tqdm(range(self.n_epochs)):
            permutation = torch.randperm(n_sample)

            for i in range(0, n_sample, self.batch_size):
                indices = permutation[i:i + self.batch_size]
                batch_A = train_t.treatment[indices]
                batch_W = train_t.outcome_proxy[indices]
                batch_Z = train_t.treatment_proxy[indices]
                batch_X = train_t.backdoor[indices]
                batch_y = train_t.outcome[indices]

                optimizer.zero_grad()
                batch_inputs = torch.cat((batch_A, batch_W, batch_X), dim=1)
                pred_y = model(batch_inputs)

                kernel_inputs_train = torch.cat((batch_A, batch_Z, batch_X), dim=1)
                kernel_matrix_train = self.compute_kernel(kernel_inputs_train)

                causal_loss_train = NMMR_loss(pred_y, batch_y, kernel_matrix_train, self.loss_name)
                causal_loss_train.backward()
                optimizer.step()

            # at the end of each epoch, log metrics
            if self.log_metrics:
                with torch.no_grad():
                    preds_train = model(torch.cat((train_t.treatment, train_t.outcome_proxy, train_t.backdoor), dim=1))
                    preds_val = model(torch.cat((val_t.treatment, val_t.outcome_proxy, val_t.backdoor), dim=1))

                    # compute the full kernel matrix
                    kernel_inputs_train = torch.cat((train_t.treatment, train_t.treatment_proxy, train_t.backdoor), dim=1)
                    kernel_inputs_val = torch.cat((val_t.treatment, val_t.treatment_proxy, val_t.backdoor), dim=1)
                    kernel_matrix_train = self.compute_kernel(kernel_inputs_train)
                    kernel_matrix_val = self.compute_kernel(kernel_inputs_val)

                    # calculate and log the causal loss (train & validation)
                    causal_loss_train = NMMR_loss(preds_train, train_t.outcome, kernel_matrix_train, self.loss_name)
                    causal_loss_val = NMMR_loss(preds_val, val_t.outcome, kernel_matrix_val, self.loss_name)
                    self.writer.add_scalar(f'{self.loss_name}/train', causal_loss_train, epoch)
                    self.writer.add_scalar(f'{self.loss_name}/val', causal_loss_val, epoch)
                    self.causal_train_losses.append(causal_loss_train)
                    self.causal_val_losses.append(causal_loss_val)

        return model


    @staticmethod
    def predict(model, test_data_t: NMMRTestDataSet_h):
        # Create a 3-dim array with shape [2, n_samples, (3 + len(X))]
        # The first axis contains the two values of do(A): 0 and 1
        # The last axis contains A, W, X, needed for the model's forward pass
        intervention_array_len = 2
        n_samples = test_data_t.outcome_proxy.shape[0]
        tempA = test_data_t.treatment.unsqueeze(-1).expand(-1, n_samples, -1)
        tempW = test_data_t.outcome_proxy.unsqueeze(0).expand(intervention_array_len, -1, -1)
        tempX = test_data_t.backdoor.unsqueeze(0).expand(intervention_array_len, -1, -1)
        model_inputs_test = torch.dstack((tempA, tempW, tempX))

        # Compute model's predicted E[Y | do(A)] = E_{w, x}[h(a, w, x)] for A in [0, 1]
        # Note: the mean is taken over the n_samples axis, so we obtain 2 avg. pot. outcomes; their diff is the ATE
        with torch.no_grad():
            E_wx_hawx = model(model_inputs_test)

        return E_wx_hawx.cpu()


class NMMR_Trainer_Experiment_q:
    def __init__(self, train_params: Dict[str, Any], random_seed: int, dump_folder: Optional[Path] = None):
        self.train_params = train_params
        self.n_epochs = train_params['n_epochs']
        self.batch_size = train_params['batch_size']
        self.gpu_flg = torch.cuda.is_available()
        self.log_metrics = train_params['log_metrics'] == "True"
        self.l2_penalty = train_params['l2_penalty']
        self.learning_rate = train_params['learning_rate']
        self.loss_name = train_params['loss_name']

        self.mse_loss = nn.MSELoss()

        if self.log_metrics:
            self.writer = SummaryWriter(log_dir=op.join(dump_folder, f"tensorboard_log_{random_seed}"))
            self.causal_train_losses = []
            self.causal_val_losses = []

    def compute_kernel(self, kernel_inputs):
        return calculate_kernel_matrix(kernel_inputs)
    
    def create(self, train_t: NMMRTrainDataSetTorch_q):
        input_size = 1 + 1 + train_t.backdoor.shape[1]
        model = MLP_for_NMMR(input_dim=input_size, train_params=self.train_params)
        return model

    def train(self, train_t: NMMRTrainDataSetTorch_q, val_t: NMMRTrainDataSetTorch_q, verbose: int = 0):
        # inputs consist of (A, W, X) tuples
        input_size = 1 + 1 + train_t.backdoor.shape[1]
        model = MLP_for_NMMR(input_dim=input_size, train_params=self.train_params)

        if self.gpu_flg:
            train_t = train_t.to_gpu()
            val_t = val_t.to_gpu()
            model.cuda()

        # weight_decay implements L2 penalty
        optimizer = optim.Adam(list(model.parameters()), lr=self.learning_rate, weight_decay=self.l2_penalty)

        n_sample = train_t.treatment.shape[0]
        # train model
        for epoch in tqdm(range(self.n_epochs)):
            permutation = torch.randperm(n_sample)

            for i in range(0, n_sample, self.batch_size):
                indices = permutation[i:i + self.batch_size]
                batch_A = train_t.treatment[indices]
                batch_tarA = train_t.treatment_target[indices]
                batch_W = train_t.outcome_proxy[indices]
                batch_Z = train_t.treatment_proxy[indices]
                batch_X = train_t.backdoor[indices]
                batch_y = train_t.outcome[indices]

                optimizer.zero_grad()
                batch_inputs = torch.cat((batch_tarA, batch_Z, batch_X), dim=1)
                pred_y = model(batch_inputs)

                # TODO: check that kernel matrix isn't too dominated by X's (vs. A and Z)
                kernel_inputs_train = torch.cat((batch_tarA, batch_W, batch_X), dim=1)
                kernel_matrix_train = self.compute_kernel(kernel_inputs_train)

                causal_loss_train = NMMR_loss(torch.mul((batch_A == batch_tarA), pred_y), torch.abs(batch_tarA), kernel_matrix_train, self.loss_name)
                causal_loss_train.backward()
                optimizer.step()

            # at the end of each epoch, log metrics
            if self.log_metrics:
                with torch.no_grad():
                    preds_train = model(torch.cat((train_t.treatment_target, train_t.treatment_proxy, train_t.backdoor), dim=1))
                    preds_val = model(torch.cat((val_t.treatment_target, val_t.treatment_proxy, val_t.backdoor), dim=1))

                    # compute the full kernel matrix
                    kernel_inputs_train = torch.cat((train_t.treatment_target, train_t.treatment_proxy, train_t.backdoor), dim=1)
                    kernel_inputs_val = torch.cat((val_t.treatment_target, val_t.treatment_proxy, val_t.backdoor), dim=1)
                    kernel_matrix_train = self.compute_kernel(kernel_inputs_train)
                    kernel_matrix_val = self.compute_kernel(kernel_inputs_val)

                    # calculate and log the causal loss (train & validation)
                    causal_loss_train = NMMR_loss(torch.mul((train_t.treatment==train_t.treatment_target), preds_train), torch.abs(train_t.treatment_target), kernel_matrix_train, self.loss_name)
                    causal_loss_val = NMMR_loss(torch.mul((val_t.treatment==val_t.treatment_target), preds_val), torch.abs(val_t.treatment_target), kernel_matrix_val, self.loss_name)
                    self.writer.add_scalar(f'{self.loss_name}/train', causal_loss_train, epoch)
                    self.writer.add_scalar(f'{self.loss_name}/val', causal_loss_val, epoch)
                    self.causal_train_losses.append(causal_loss_train)
                    self.causal_val_losses.append(causal_loss_val)

        return model


    @staticmethod
    def predict(model, test_data_t: NMMRTestDataSet_q):
        # Create a 3-dim array with shape [2, n_samples, (3 + len(X))]
        # The first axis contains the two values of do(A): 0 and 1
        # The last axis contains A, W, X, needed for the model's forward pass
        intervention_array_len = 1
        n_samples = test_data_t.treatment_proxy.shape[0]
        tempA = test_data_t.treatment.unsqueeze(-1).expand(-1, n_samples, -1)
        tempZ = test_data_t.treatment_proxy.unsqueeze(0).expand(intervention_array_len, -1, -1)
        tempX = test_data_t.backdoor.unsqueeze(0).expand(intervention_array_len, -1, -1)
        model_inputs_test = torch.dstack((tempA, tempZ, tempX))

        # Compute model's predicted E[Y | do(A)] = E_{w, x}[h(a, w, x)] for A in [0, 1]
        # Note: the mean is taken over the n_samples axis, so we obtain 2 avg. pot. outcomes; their diff is the ATE
        with torch.no_grad():
            E_zx_qazx = model(model_inputs_test)

        return E_zx_qazx.cpu()
