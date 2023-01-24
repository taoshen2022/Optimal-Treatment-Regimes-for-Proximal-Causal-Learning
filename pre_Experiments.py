import os
import json
import click
import torch
from pathlib import Path
from shutil import make_archive
import os
import datetime
import numpy as np
import random
import math

from data.DGP import DGP, DGP_test
from data.DGP import DGP
from src.data_type.data_class import NMMRTrainDataSet_h, NMMRTrainDataSetTorch_h, NMMRTrainDataSet_q, NMMRTrainDataSetTorch_q, NMMRTestDataSetTorch_h, NMMRTestDataSet_h, NMMRTestDataSetTorch_h, NMMRTestDataSet_q, NMMRTestDataSetTorch_q
from src.models.NMMR_trainers import NMMR_Trainer_Experiment_h, NMMR_Trainer_Experiment_q
from src.models.NMMR_experiments import NMMR_experiment_h, NMMR_experiment_q
import random

###### Extract values for h, q0, q1 estimation for ITR estimation ########

os.chdir(os.getcwd())

DATA_DIR = Path.cwd().joinpath('data')

config_path_h = Path.cwd().joinpath('configs/nmmr_finalized_h.json')
config_path_q = Path.cwd().joinpath('configs/nmmr_finalized_q.json')

with open(config_path_h) as f:
        config_h = json.load(f)
with open(config_path_q) as f:
        config_q = json.load(f)

model_name = config_h['model']['name']
foldername = str(datetime.datetime.now().strftime("%m-%d-%H-%M-%S"))

data_config = config_h["data"]
model_config_h = config_h["model"]
model_config_q = config_q["model"]

data_dir = Path.cwd().joinpath('data')

######### Prepare for test data ##############
random.seed(12)
n_test = 10000
n_train = 1000
p_train = 2
n_sim = 200
n_sc = 6
X_t, Z_t, W_t, U_t = DGP_test(n_test)
one_dump_dir = data_dir.joinpath('test case')
os.mkdir(one_dump_dir)
one_dump_dir = str(one_dump_dir)
np.save(one_dump_dir+'/X_t.npy', X_t)
np.save(one_dump_dir+'/Z_t.npy', Z_t)
np.save(one_dump_dir+'/W_t.npy', W_t)
np.save(one_dump_dir+'/U_t.npy', U_t)

#X_t = np.load(one_dump_dir+'/X_t.npy')
#Z_t = np.load(one_dump_dir+'/Z_t.npy')
#W_t = np.load(one_dump_dir+'/W_t.npy')
#U_t = np.load(one_dump_dir+'/U_t.npy')

one_dump_dir = data_dir.joinpath('one')
one_mdl_dump_dir = one_dump_dir.joinpath('l')

########## NMMR Estimation under each scenario (including training data generation) ##############
for s_num in range(n_sc):
    one_dump_dir = data_dir.joinpath('S'+str(s_num+1))
    os.mkdir(one_dump_dir)
    dir_test = data_dir.joinpath('test case')
    dir_test = dir_test.joinpath('S'+str(s_num+1))
    os.mkdir(dir_test) 
    for i in range(n_sim):
      one_dump_dir = data_dir.joinpath('S'+str(s_num+1)).joinpath('repeat'+str(i))
      os.mkdir(one_dump_dir)
      one_dump_dir = str(one_dump_dir)
      random.seed(i)
      X, Z, W, U, A, Y = DGP(n_train,s_num+1)
      np.save(one_dump_dir+'/X.npy', X)
      np.save(one_dump_dir+'/Z.npy', Z)
      np.save(one_dump_dir+'/W.npy', W)
      np.save(one_dump_dir+'/A.npy', A)
      np.save(one_dump_dir+'/Y.npy', Y)

      ################## h estimation for training set #############
      train_data = NMMRTrainDataSet_h(treatment=A, treatment_proxy=Z,outcome_proxy=W,outcome=Y, backdoor=X)
      train_torch = NMMRTrainDataSetTorch_h.from_numpy(train_data)
      val_data = NMMRTrainDataSet_h(treatment=A, treatment_proxy=Z,outcome_proxy=W,outcome=Y, backdoor=X)
      val_torch = NMMRTrainDataSetTorch_h.from_numpy(val_data)

      trainer = NMMR_Trainer_Experiment_h(model_config_h, i, one_mdl_dump_dir)

      model = trainer.train(train_torch, val_torch, 0)
      test_data = NMMRTestDataSet_h(treatment=np.array([[-1], [1]]),
                                  outcome_proxy=W,
                                  backdoor=X)
      test_torch = NMMRTestDataSetTorch_h.from_numpy(test_data)
      if trainer.gpu_flg:
          test_torch = test_torch.to_gpu()

      E_wx_hawx = trainer.predict(model, test_torch)
      pred = E_wx_hawx.detach().numpy()
      np.save(one_dump_dir+'/h.npy', pred)
      torch.save(model.state_dict(), one_dump_dir+'/model_h.pth')

      ################## h estimation for delta estimation #############
      #dir_test = data_dir.joinpath('test case').joinpath('S'+str(s_num+1)).joinpath('repeat'+str(i))
      #os.mkdir(dir_test)
      #dir_test = str(dir_test)
      #for j in range(n_test):
        #X_b = np.zeros((n_train,p_train))
        #for k in range(n_train):
          #X_b[k,:] = X_t[j,:]
        #test_data = NMMRTestDataSet_h(treatment=np.array([[-1], [1]]),
                                  #outcome_proxy=W,
                                  #backdoor=X_b)
        #test_torch = NMMRTestDataSetTorch_h.from_numpy(test_data)
        #if trainer.gpu_flg:
            #test_torch = test_torch.to_gpu()

        #E_wx_hawx = trainer.predict(model, test_torch)
        #pred = E_wx_hawx.detach().numpy()
        #np.save(dir_test+'/h_for_'+str(j)+'.npy', pred)


      ################## q1 estimation for training set #############
      trainer = NMMR_Trainer_Experiment_q(model_config_q, i, one_mdl_dump_dir)

      A2 = np.ones((n_train,1))
      train_data = NMMRTrainDataSet_q(treatment=A, treatment_target = A2, treatment_proxy=Z,outcome_proxy=W,outcome=Y,backdoor=X)
      train_torch = NMMRTrainDataSetTorch_q.from_numpy(train_data)
      val_data = NMMRTrainDataSet_q(treatment=A, treatment_target = A2, treatment_proxy=Z,outcome_proxy=W,outcome=Y,backdoor=X)
      val_torch = NMMRTrainDataSetTorch_q.from_numpy(val_data)
      model = trainer.train(train_torch, val_torch, 0)
      test_data = NMMRTestDataSet_q(treatment=np.array([[1]]),
                                  treatment_proxy=Z,
                                  backdoor=X)
      test_torch = NMMRTestDataSetTorch_q.from_numpy(test_data)
      if trainer.gpu_flg:
          test_torch = test_torch.to_gpu()

      E_zx_qazx = trainer.predict(model, test_torch)
      pred = E_zx_qazx.detach().numpy()
      np.save(one_dump_dir+'/q1.npy', pred)
      torch.save(model.state_dict(), one_dump_dir+'/model_q1.pth')

      ################## q1 estimation for delta estimation #############
      #for j in range(n_test):
        #X_b = np.zeros((n_train,p_train))
        #for k in range(n_train):
          #X_b[k,:] = X_t[j,:]
        #test_data = NMMRTestDataSet_q(treatment=np.array([[1]]),
                                  #treatment_proxy=Z,
                                  #backdoor=X_b)
        #test_torch = NMMRTestDataSetTorch_q.from_numpy(test_data)
        #if trainer.gpu_flg:
            #test_torch = test_torch.to_gpu()

        #E_wx_hawx = trainer.predict(model, test_torch)
        #pred = E_wx_hawx.detach().numpy()
        #np.save(dir_test+'/q1_for_'+str(j)+'.npy', pred)

      ################## q0 estimation for training set #############
      trainer = NMMR_Trainer_Experiment_q(model_config_q, i, one_mdl_dump_dir)

      A2 = -np.ones((n_train,1))
      train_data = NMMRTrainDataSet_q(treatment=A, treatment_target = A2, treatment_proxy=Z,outcome_proxy=W,outcome=Y,backdoor=X)
      train_torch = NMMRTrainDataSetTorch_q.from_numpy(train_data)
      val_data = NMMRTrainDataSet_q(treatment=A, treatment_target = A2, treatment_proxy=Z,outcome_proxy=W,outcome=Y,backdoor=X)
      val_torch = NMMRTrainDataSetTorch_q.from_numpy(val_data)
      model = trainer.train(train_torch, val_torch, 0)
      test_data = NMMRTestDataSet_q(treatment=np.array([[-1]]),
                                  treatment_proxy=Z,
                                  backdoor=X)
      test_torch = NMMRTestDataSetTorch_q.from_numpy(test_data)
      if trainer.gpu_flg:
          test_torch = test_torch.to_gpu()

      E_zx_qazx = trainer.predict(model, test_torch)
      pred = E_zx_qazx.detach().numpy()
      np.save(one_dump_dir+'/q0.npy', pred)
      torch.save(model.state_dict(), one_dump_dir+'/model_q0.pth')

      ################## q0 estimation for delta estimation #############
      #for j in range(n_test):
        #X_b = np.zeros((n_train,p_train))
        #for k in range(n_train):
          #X_b[k,:] = X_t[j,:]
        #test_data = NMMRTestDataSet_q(treatment=np.array([[-1]]),
                                  #treatment_proxy=Z,
                                  #backdoor=X_b)
        #test_torch = NMMRTestDataSetTorch_q.from_numpy(test_data)
        #if trainer.gpu_flg:
            #test_torch = test_torch.to_gpu()

        #E_wx_hawx = trainer.predict(model, test_torch)
        #pred = E_wx_hawx.detach().numpy()
        #np.save(dir_test+'/q0_for_'+str(j)+'.npy', pred)
