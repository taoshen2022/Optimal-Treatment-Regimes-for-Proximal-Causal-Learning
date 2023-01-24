from src.models.kernel_utils import gaussian_kernel
import numpy as np
from data.DGP import DGP_eval
from src.models.pre_ITR import pre_ITR_estimation
import os 
from pathlib import Path
import json
from src.models.NMMR_trainers import NMMR_Trainer_Experiment_h, NMMR_Trainer_Experiment_q
from src.data_type.data_class import NMMRTrainDataSet_h, NMMRTrainDataSetTorch_h, NMMRTrainDataSet_q, NMMRTrainDataSetTorch_q, NMMRTestDataSetTorch_h, NMMRTestDataSet_h, NMMRTestDataSetTorch_h, NMMRTestDataSet_q, NMMRTestDataSetTorch_q
import datetime
import torch

def con_h(DATA_DIR, s_num, rep_num, n_train, p_train, X_t, tar_index):

    #params:
    #DATA_DIR: directory for the stored model aftering pre_experiment
    #s_sum: scenario number
    #rep_num: number of repition for simulation
    #n_train: number of records in training data
    #p_train: dimension in X
    #tar_index: the index of current record for evaluation

    ####estimating h(w,a,x) t0 construct delta#######

    model_h = torch.load(DATA_DIR.joinpath("S"+str(s_num)+"/repeat"+str(rep_num)+"/model_h.pth"))
    config_path_h = Path.cwd().joinpath('configs/nmmr_finalized_h.json')

    with open(config_path_h) as f:
        config_h = json.load(f)

    model_config_h = config_h["model"]
    one_dump_dir = DATA_DIR.joinpath('one')
    one_mdl_dump_dir = one_dump_dir.joinpath('l')

    trainer_h = NMMR_Trainer_Experiment_h(model_config_h, 0, one_mdl_dump_dir)
    X, Z, W, A, Y =  np.load(DATA_DIR.joinpath("S"+str(s_num)+"/repeat"+str(rep_num)+"/X.npy")), np.load(DATA_DIR.joinpath("S"+str(s_num)+"/repeat"+str(rep_num)+"/Z.npy")), np.load(DATA_DIR.joinpath("S"+str(s_num)+"/repeat"+str(rep_num)+"/W.npy")), np.load(DATA_DIR.joinpath("S"+str(s_num)+"/repeat"+str(rep_num)+"/A.npy")), np.load(DATA_DIR.joinpath("S"+str(s_num)+"/repeat"+str(rep_num)+"/Y.npy"))
    train_data_h = NMMRTrainDataSet_h(treatment=A, treatment_proxy=Z,outcome_proxy=W,outcome=Y, backdoor=X)
    model_h_fin = trainer_h.create(train_data_h)
    model_h_fin.load_state_dict(model_h)

    X_b = np.zeros((n_train,p_train))
    for k in range(n_train):
        X_b[k,:] = X_t[tar_index,:]
    test_data = NMMRTestDataSet_h(treatment=np.array([[-1], [1]]),
                                outcome_proxy=W,
                                backdoor=X_b)
    test_torch = NMMRTestDataSetTorch_h.from_numpy(test_data)
    if trainer_h.gpu_flg:
        test_torch = test_torch.to_gpu()

    E_wx_hawx = trainer_h.predict(model_h_fin, test_torch)
    pred = E_wx_hawx.detach().numpy()
    return pred

def con_q1(DATA_DIR, s_num, rep_num, n_train, p_train, X_t, tar_index):

    #params:
    #DATA_DIR: directory for the stored model aftering pre_experiment
    #s_sum: scenario number
    #rep_num: number of repition for simulation
    #n_train: number of records in training data
    #p_train: dimension in X
    #tar_index: the index of current record for evaluation

     ####estimating q(z,1,x) t0 construct delta#######

    model_q1 = torch.load(DATA_DIR.joinpath("S"+str(s_num)+"/repeat"+str(rep_num)+"/model_q1.pth"))
    config_path_q = Path.cwd().joinpath('configs/nmmr_finalized_q.json')

    with open(config_path_q) as f:
        config_q = json.load(f)

    model_config_q = config_q["model"]
    one_dump_dir = DATA_DIR.joinpath('one')
    one_mdl_dump_dir = one_dump_dir.joinpath('l')

    trainer_q1 = NMMR_Trainer_Experiment_q(model_config_q, 0, one_mdl_dump_dir)
    X, Z, W, A, Y =  np.load(DATA_DIR.joinpath("S"+str(s_num)+"/repeat"+str(rep_num)+"/X.npy")), np.load(DATA_DIR.joinpath("S"+str(s_num)+"/repeat"+str(rep_num)+"/Z.npy")), np.load(DATA_DIR.joinpath("S"+str(s_num)+"/repeat"+str(rep_num)+"/W.npy")), np.load(DATA_DIR.joinpath("S"+str(s_num)+"/repeat"+str(rep_num)+"/A.npy")), np.load(DATA_DIR.joinpath("S"+str(s_num)+"/repeat"+str(rep_num)+"/Y.npy"))
    A2 = np.ones((n_train,1))
    train_data_q1 = NMMRTrainDataSet_q(treatment=A, treatment_target = A2, treatment_proxy=Z,outcome_proxy=W,outcome=Y,backdoor=X)
    model_q1_fin = trainer_q1.create(train_data_q1)
    model_q1_fin.load_state_dict(model_q1)

    X_b = np.zeros((n_train,p_train))
    for k in range(n_train):
        X_b[k,:] = X_t[tar_index,:]

    test_data = NMMRTestDataSet_q(treatment=np.array([[1]]),
                                  treatment_proxy=Z,
                                  backdoor=X_b)
    test_torch = NMMRTestDataSetTorch_q.from_numpy(test_data)
    if trainer_q1.gpu_flg:
        test_torch = test_torch.to_gpu()

    E_zx_qazx = trainer_q1.predict(model_q1_fin, test_torch)
    pred = E_zx_qazx.detach().numpy()
    return pred

def con_q0(DATA_DIR, s_num, rep_num, n_train, p_train, X_t, tar_index):

    #params:
    #DATA_DIR: directory for the stored model aftering pre_experiment
    #s_sum: scenario number
    #rep_num: number of repition for simulation
    #n_train: number of records in training data
    #p_train: dimension in X
    #tar_index: the index of current record for evaluation

    ####estimating q(z,-1,x) t0 construct delta#######

    model_q0 = torch.load(DATA_DIR.joinpath("S"+str(s_num)+"/repeat"+str(rep_num)+"/model_q0.pth"))
    config_path_q = Path.cwd().joinpath('configs/nmmr_finalized_q.json')

    with open(config_path_q) as f:
        config_q = json.load(f)

    model_config_q = config_q["model"]
    one_dump_dir = DATA_DIR.joinpath('one')
    one_mdl_dump_dir = one_dump_dir.joinpath('l')

    trainer_q0 = NMMR_Trainer_Experiment_q(model_config_q, 0, one_mdl_dump_dir)
    X, Z, W, A, Y =  np.load(DATA_DIR.joinpath("S"+str(s_num)+"/repeat"+str(rep_num)+"/X.npy")), np.load(DATA_DIR.joinpath("S"+str(s_num)+"/repeat"+str(rep_num)+"/Z.npy")), np.load(DATA_DIR.joinpath("S"+str(s_num)+"/repeat"+str(rep_num)+"/W.npy")), np.load(DATA_DIR.joinpath("S"+str(s_num)+"/repeat"+str(rep_num)+"/A.npy")), np.load(DATA_DIR.joinpath("S"+str(s_num)+"/repeat"+str(rep_num)+"/Y.npy"))
    A2 = -np.ones((n_train,1))
    train_data_q0 = NMMRTrainDataSet_q(treatment=A, treatment_target = A2, treatment_proxy=Z,outcome_proxy=W,outcome=Y,backdoor=X)
    model_q0_fin = trainer_q0.create(train_data_q0)
    model_q0_fin.load_state_dict(model_q0)

    X_b = np.zeros((n_train,p_train))
    for k in range(n_train):
        X_b[k,:] = X_t[tar_index,:]

    test_data = NMMRTestDataSet_q(treatment=np.array([[-1]]),
                                  treatment_proxy=Z,
                                  backdoor=X_b)
    test_torch = NMMRTestDataSetTorch_q.from_numpy(test_data)
    if trainer_q0.gpu_flg:
        test_torch = test_torch.to_gpu()

    E_zx_qazx = trainer_q0.predict(model_q0_fin, test_torch)
    pred = E_zx_qazx.detach().numpy()
    return pred


def ITR_experiment(s_num, rep_num):

    #params:
    #s_sum: scenario number
    #rep_num: number of repition for simulation

    os.chdir(os.getcwd())
    DATA_DIR = Path.cwd().joinpath('data')

    ###################Data generation################
    X_t, Z_t, W_t, U_t = np.load(DATA_DIR.joinpath('test case/X_t.npy')), np.load(DATA_DIR.joinpath('test case/Z_t.npy')), np.load(DATA_DIR.joinpath('test case/W_t.npy')), np.load(DATA_DIR.joinpath('test case/U_t.npy'))
    X, Z, W, A, Y =  np.load(DATA_DIR.joinpath("S"+str(s_num)+"/repeat"+str(rep_num)+"/X.npy")), np.load(DATA_DIR.joinpath("S"+str(s_num)+"/repeat"+str(rep_num)+"/Z.npy")), np.load(DATA_DIR.joinpath("S"+str(s_num)+"/repeat"+str(rep_num)+"/W.npy")), np.load(DATA_DIR.joinpath("S"+str(s_num)+"/repeat"+str(rep_num)+"/A.npy")), np.load(DATA_DIR.joinpath("S"+str(s_num)+"/repeat"+str(rep_num)+"/Y.npy"))
    ##################pre ITR computation ################
    path_h, path_q1, path_q0 = DATA_DIR.joinpath("S"+str(s_num)+"/repeat"+str(rep_num)+"/h.npy"), DATA_DIR.joinpath("S"+str(s_num)+"/repeat"+str(rep_num)+"/q1.npy"), DATA_DIR.joinpath("S"+str(s_num)+"/repeat"+str(rep_num)+"/q0.npy")
    config_path = Path.cwd().joinpath('configs/pre_ITR_selection.json')
    with open(config_path) as f:
        config = json.load(f)


    pena_list = config['model']['penalty']
    batch_size = config['model']['cv_size']
    a1, a2 = pre_ITR_estimation(pena_list, batch_size, X,Z,W,A,Y,path_h,path_q1,path_q0)
    n_x = len(Z_t)
    #rez = a1[0] * X_t[:,0].reshape(n_x,1) + a1[1] * X_t[:,1].reshape(n_x,1) + a1[2] * Z_t + a1[3]
    rez = a1[0] * X_t[:,0] + a1[1] * X_t[:,1] + a1[2] * Z_t + a1[3] * X_t[:,0] * Z_t + a1[4] * X_t[:,1] * Z_t
    A_tz = np.ones((len(rez),1))
    for i in range(len(rez)):
        if rez[i][0] >= 0:
            A_tz[i,0] = 1
        else:
            A_tz[i,0] = -1
    Y_tz = DGP_eval(10000, s_num, A_tz, X_t, W_t, Z_t, U_t)

    #rew = a2[0] * X_t[:,0].reshape(n_x,1) + a2[1] * X_t[:,1].reshape(n_x,1) + a2[2] * W_t + a2[3]
    rew  =  a2[0] * X_t[:,0] + a2[1] * X_t[:,1] + a2[2] * W_t + a2[3] * X_t[:,0] * W_t + a2[4] * X_t[:,1] * W_t
    A_tw = np.ones((len(rew),1))
    for i in range(len(rew)):
        if rew[i][0] >= 0:
            A_tw[i,0] = 1
        else:
            A_tw[i,0] = -1
    Y_tw = DGP_eval(10000, s_num, A_tw, X_t, W_t, Z_t, U_t)

    ########### z cup w in qi ################
    #oriz = a1[0] * X[:,0].reshape(len(Z),1) + a1[1] * X[:,1].reshape(len(Z),1) + a1[2] * Z + a1[3] 
    #oriw = a2[0] * X[:,0].reshape(len(Z),1) + a2[1] * X[:,1].reshape(len(Z),1) + a2[2] * W + a2[3] 
    oriz = a1[0] * X[:,0] + a1[1] * X[:,1] + a1[2] * Z + a1[3] * X[:,0] * Z + a1[4] * X[:,1] * Z
    oriw = a2[0] * X[:,0] + a2[1] * X[:,1] + a2[2] * W + a2[3] * X[:,0] * Z + a1[4] * X[:,1] * Z
    


    ############### h computation for a = 1 and -1 ############
    h_re = np.load(path_h)
    h = h_re[1] * (oriz >= 0) + h_re[0] * (oriz < 0)


    ############### q computation for a = 1 and -1 ############
    q1_re, q0_re = np.load(path_q1), np.load(path_q0)
    q = Y * ((A == 1) * q1_re[0] + (A == -1) * q0_re[0]) * (A == 1) * (oriw >= 0) + Y * ((A == 1) * q1_re[0] + (A == -1) * q0_re[0]) * (A == -1) * (oriw < 0)

    if np.mean(h) >= np.mean(q):
        Y_tzcw = Y_tz
    else:
        Y_tzcw = Y_tw

    ####################The proposed estimator #################
    A_tzw = np.ones((len(rew),1))

    for i in range(len(rew)):
        if A_tz[i] == A_tw[i]:
            A_tzw[i,0] = A_tz[i]
        else:
            ht_re = con_h(DATA_DIR, s_num, rep_num, len(Z), len(X[0,:]), X_t, i)
            #ht_re = np.load(DATA_DIR.joinpath("test case/S"+str(s_num)+"/repeat"+str(rep_num)+"/h_for_"+str(i)+".npy"))
            hz = ht_re[1] * (A_tz[i] == 1) +  ht_re[0] * (A_tz[i] == -1)

            q1t_re, q0t_re = con_q1(DATA_DIR, s_num, rep_num, len(Z), len(X[0,:]), X_t, i), con_q0(DATA_DIR, s_num, rep_num, len(Z), len(X[0,:]), X_t, i)
            #q1t_re = np.load(DATA_DIR.joinpath("test case/S"+str(s_num)+"/repeat"+str(rep_num)+"/q1_for_"+str(i)+".npy"))
            #q0t_re = np.load(DATA_DIR.joinpath("test case/S"+str(s_num)+"/repeat"+str(rep_num)+"/q0_for_"+str(i)+".npy"))
            qw = (Y * ((A == 1) * q1t_re[0] + (A == -1) * q0t_re[0]) * (A == 1) * (A_tw[i] == 1)) +  (Y * ((A == 1) * q1t_re[0] + (A == -1) * q0t_re[0]) * (A == -1) * (A_tw[i] == -1))

            band = 1.06 * 500**(-0.2) * np.std(X[:,0])
            x_ker = gaussian_kernel(X, X_t[i,:], band)
            if np.sum(hz * x_ker)/np.sum(x_ker) >= np.sum((qw+2) * x_ker)/np.sum(x_ker):
                A_tzw[i,0] = A_tz[i,0]
            else:
                A_tzw[i,0] = A_tw[i,0]
    Y_tzw = DGP_eval(10000, s_num, A_tzw, X_t, W_t, Z_t, U_t)
    print("Results for scenario"+str(s_num)+":"+str(np.mean(Y_tz))+","+str(np.mean(Y_tw))+","+str(np.mean(Y_tzcw))+","+str(np.mean(Y_tzw)))

    return [s_num, np.mean(Y_tz), np.mean(Y_tw), np.mean(Y_tzcw), np.mean(Y_tzw)]





