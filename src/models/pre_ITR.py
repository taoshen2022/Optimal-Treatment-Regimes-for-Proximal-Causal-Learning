import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
import time
import os
import json
import torch
from pathlib import Path
from src.data_type.data_class import NMMRTrainDataSet_h, NMMRTrainDataSetTorch_h, NMMRTrainDataSet_q, NMMRTrainDataSetTorch_q, NMMRTestDataSet_h, NMMRTestDataSetTorch_h, NMMRTestDataSet_q, NMMRTestDataSetTorch_q
from src.models.NMMR_trainers import NMMR_Trainer_Experiment_h, NMMR_Trainer_Experiment_q
    

def pre_ITR_estimation(pena_list, batch_size, X, Z, W, A, Y, path_h, path_q1, path_q0):
    #params:
    #X, Z, W, A, Y: covariates from training data

    h_re = np.load(path_h)
    q1_re, q0_re = np.load(path_q1), np.load(path_q0)

    ########## rho_z, rho_w selection ################
    val_h, val_q = [], []
    for rho in pena_list:
        h_est, q_est = 0, 0
        for k in range(int(len(Z)/batch_size)):
            permutation = torch.randperm(len(Z))
            val_indices = permutation[0:batch_size]
            train_indices = permutation[batch_size:len(Z)]

            X_V, Z_V, W_V, A_V, Y_V = X[train_indices,:], Z[train_indices], W[train_indices], A[train_indices], Y[train_indices]
            X_T, Z_T, W_T, A_T, Y_T = X[val_indices,:], Z[val_indices], W[val_indices], A[val_indices], Y[val_indices]
            h1_V, h0_V, q1_V, q0_V = h_re[1][train_indices], h_re[0][train_indices], q1_re[0][train_indices], q0_re[0][train_indices]
            h1_T, h0_T, q1_T, q0_T = h_re[1][val_indices], h_re[0][val_indices], q1_re[0][val_indices], q0_re[0][val_indices]

            abs_v, sign_v = abs(h1_V - h0_V),  (2*((h1_V - h0_V)>=0)-1)
            n, p = len(X_V[:,1]), len(X_V[0,:])
            #A_s = np.concatenate((X_V,Z_V,np.ones((n,1))), axis = 1)
            A_s = np.concatenate((X_V,Z_V,np.multiply(X_V[:,0], Z_V), np.multiply(X_V[:,1],Z_V)), axis = 1)
            def xz_obj(x):
                obj = 0
                for i in range(n):
                    #obj += abs_v[i] * max(1-(sign_v[i]*(A_s[i,0]*x[0]+A_s[i,1]*x[1]+A_s[i,2]*x[2]+A_s[i,3]*x[3])),0)
                    obj += abs_v[i] * max(1-(sign_v[i]*(A_s[i,0]*x[0]+A_s[i,1]*x[1]+A_s[i,2]*x[2]+A_s[i,3]*x[3]+A_s[i,4]*x[4])),0)
                return obj/n+rho*(np.sum(x**2))
            def xz_obj_der(x):
                obj = 0
                for i in range(n):
                    #obj += abs_v[i] * (-(sign_v[i]*[A_s[i,0], A_s[i,1], A_s[i,2],A_s[i,3]])) * ((1-sign_v[i]*(A_s[i,0]*x[0]+A_s[i,1]*x[1]+A_s[i,2]*x[2]+A_s[i,3]*x[3]))>=0)
                    obj += abs_v[i] * (-(sign_v[i]*[A_s[i,0], A_s[i,1], A_s[i,2],A_s[i,3],A_s[i,4]])) * ((1-sign_v[i]*(A_s[i,0]*x[0]+A_s[i,1]*x[1]+A_s[i,2]*x[2]+A_s[i,3]*x[3]+A_s[i,4]*x[4]))>=0)
                return obj/n+rho*x
            x0 = np.random.randint(-100,100,5) * 0.01
            res = minimize(xz_obj, x0, method = 'BFGS', jac = xz_obj_der, options = {'disp':True})
            a1 = res.x

            #d_bz = a1[0] * X_T[:,0].reshape(len(Z_T),1) + a1[1] * X_T[:,1].reshape(len(Z_T),1) + a1[2] * Z_T + a1[3] 
            d_bz = a1[0] * X_T[:,0] + a1[1] * X_T[:,1] + a1[2] * Z_T + a1[3]  * X_T[:,0] * Z_T + a1[4] * X_T[:,1] * Z_T
            h_est += np.mean(h1_T * (d_bz>=0) + h0_T * (d_bz<0))

            q_V = Y_V * ((A_V == 1) * q1_V + (A_V == -1) * q0_V) * (A_V == 1) - Y_V * ((A_V == 1) * q1_V + (A_V == -1) * q0_V) * (A_V == -1)
            abs_v, sign_v = abs(q_V), 2*(q_V>=0)-1
            #A_s = np.concatenate((X_V,W_V,np.ones((n,1))), axis = 1)
            A_s = np.concatenate((X_V,W_V,np.multiply(X_V[:,0], W_V), np.multiply(X_V[:,1],W_V)), axis = 1)
            def xw_obj(x):
                obj = 0
                for i in range(n):
                    #obj += abs_v[i] * max(1-(sign_v[i]*(A_s[i,0]*x[0]+A_s[i,1]*x[1]+A_s[i,2]*x[2]+A_s[i,3]*x[3])),0)
                    obj += abs_v[i] * max(1-(sign_v[i]*(A_s[i,0]*x[0]+A_s[i,1]*x[1]+A_s[i,2]*x[2]+A_s[i,3]*x[3]+A_s[i,4]*x[4])),0)
                return obj/n+rho*(np.sum(x**2))
            def xw_obj_der(x):
                obj = 0
                for i in range(n):
                    #obj += abs_v[i] * (-(sign_v[i]*[A_s[i,0], A_s[i,1], A_s[i,2],A_s[i,3]])) * ((1-sign_v[i]*(A_s[i,0]*x[0]+A_s[i,1]*x[1]+A_s[i,2]*x[2]+A_s[i,3]*x[3]))>=0)
                    obj += abs_v[i] * (-(sign_v[i]*[A_s[i,0], A_s[i,1], A_s[i,2],A_s[i,3],A_s[i,4]])) * ((1-sign_v[i]*(A_s[i,0]*x[0]+A_s[i,1]*x[1]+A_s[i,2]*x[2]+A_s[i,3]*x[3]+A_s[i,4]*x[4]))>=0)
                return obj/n+rho*x
            x0 =  np.random.randint(-100,100,5) * 0.01
            res2 = minimize(xw_obj, x0, method = 'BFGS', jac = xw_obj_der, options = {'disp':True})
            a2 = res2.x

            #d_bw = a2[0] * X_T[:,0].reshape(len(Z_T),1) + a2[1] * X_T[:,1].reshape(len(Z_T),1) + a2[2] * W_T + a2[3]
            d_bw = a2[0] * X_T[:,0] + a2[1] * X_T[:,1] + a2[2] * W_T + a2[3] * X_T[:,0] * W_T + a2[4] * X_T[:,1] * W_T
            q_est +=  np.mean(Y_T * ((A_T == 1) * q1_T + (A_T == -1) * q0_T) * (A_T == 1) * (d_bw >= 0) + Y_T * ((A_T == 1) * q1_T + (A_T == -1) * q0_T) * (A_T == -1) * (d_bw < 0))
                
        val_h.append(h_est/(len(Z)/batch_size))
        val_q.append(q_est/(len(Z)/batch_size))

    rho_re_z, rho_re_w = pena_list[np.argmax(val_h)], pena_list[np.argmax(val_q)]

    ############## final h estimation ###################
    abs_v, sign_v = abs(h_re[1] - h_re[0]),  (2*((h_re[1] - h_re[0])>=0)-1)
    #############objective construction ###################
    n, p = len(X[:,1]), len(X[0,:])
    #A_s = np.concatenate((X,Z,np.ones((n,1))), axis = 1)
    A_s = np.concatenate((X,Z,np.multiply(X[:,0], Z), np.multiply(X[:,1],Z)), axis = 1)
    def xz_obj(x):
        obj = 0
        for i in range(n):
            #obj += abs_v[i] * max(1-(sign_v[i]*(A_s[i,0]*x[0]+A_s[i,1]*x[1]+A_s[i,2]*x[2]+A_s[i,3]*x[3])),0)
            obj += abs_v[i] * max(1-(sign_v[i]*(A_s[i,0]*x[0]+A_s[i,1]*x[1]+A_s[i,2]*x[2]+A_s[i,3]*x[3]+A_s[i,4]*x[4])),0)
        return obj/n+rho_re_z*(np.sum(x**2))
    def xz_obj_der(x):
        obj = 0
        for i in range(n):
            #obj += abs_v[i] * (-(sign_v[i]*[A_s[i,0], A_s[i,1], A_s[i,2],A_s[i,3]])) * ((1-sign_v[i]*(A_s[i,0]*x[0]+A_s[i,1]*x[1]+A_s[i,2]*x[2]+A_s[i,3]*x[3]))>=0)
            obj += abs_v[i] * (-(sign_v[i]*[A_s[i,0], A_s[i,1], A_s[i,2],A_s[i,3],A_s[i,4]])) * ((1-sign_v[i]*(A_s[i,0]*x[0]+A_s[i,1]*x[1]+A_s[i,2]*x[2]+A_s[i,3]*x[3]+A_s[i,4]*x[4]))>=0)
        return obj/n+rho_re_z*x
    x0 = np.random.randint(-100,100,5) * 0.01
    res = minimize(xz_obj, x0, method = 'BFGS', jac = xz_obj_der, options = {'disp':True})
    #res =  minimize(xz_obj, x0, method= 'nelder-mead', options = {'xatol':1e-8, 'disp':True})
    re_alpha = res.x    
    ############## q estimation ###################
    q_re = Y * ((A == 1) * q1_re[0] + (A == -1) * q0_re[0]) * (A == 1) - Y * ((A == 1) * q1_re[0] + (A == -1) * q0_re[0]) * (A == -1)
    abs_v, sign_v = abs(q_re), 2*(q_re>=0)-1
    n, p = len(X[:,1]), len(X[0,:])
    #A_s = np.concatenate((X,W,np.ones((n,1))), axis = 1)
    A_s = np.concatenate((X,W,np.multiply(X[:,0], W), np.multiply(X[:,1],W)), axis = 1)
    def xw_obj(x):
        obj = 0
        for i in range(n):
            #obj += abs_v[i] * max(1-(sign_v[i]*(A_s[i,0]*x[0]+A_s[i,1]*x[1]+A_s[i,2]*x[2]+A_s[i,3]*x[3])),0)
            obj += abs_v[i] * max(1-(sign_v[i]*(A_s[i,0]*x[0]+A_s[i,1]*x[1]+A_s[i,2]*x[2]+A_s[i,3]*x[3]+A_s[i,4]*x[4])),0)
        return obj/n+rho_re_w*(np.sum(x**2))
    def xw_obj_der(x):
        obj = 0
        for i in range(n):
            #obj += abs_v[i] * (-(sign_v[i]*[A_s[i,0], A_s[i,1], A_s[i,2],A_s[i,3]])) * ((1-sign_v[i]*(A_s[i,0]*x[0]+A_s[i,1]*x[1]+A_s[i,2]*x[2]+A_s[i,3]*x[3]))>=0)
            obj += abs_v[i] * (-(sign_v[i]*[A_s[i,0], A_s[i,1], A_s[i,2],A_s[i,3],A_s[i,4]])) * ((1-sign_v[i]*(A_s[i,0]*x[0]+A_s[i,1]*x[1]+A_s[i,2]*x[2]+A_s[i,3]*x[3]+A_s[i,4]*x[4]))>=0)
        return obj/n+rho_re_w*x
    x0 =  np.random.randint(-100,100,5) * 0.01
    res2 = minimize(xw_obj, x0, method = 'BFGS', jac = xw_obj_der, options = {'disp':True})
    #res2 = minimize(xw_obj, x0, method= 'nelder-mead', options = {'xatol':1e-8, 'disp':True})
    re_alpha2 = res2.x
    
    return re_alpha, re_alpha2

#X, Z, W, A, Y = np.load("C:/Users/shent/Desktop/Optimal-Individualized-Decision-Making-with-Proxies/data/S1/repeat0/X.npy"), np.load("C:/Users/shent/Desktop/Optimal-Individualized-Decision-Making-with-Proxies/data/S1/repeat0/Z.npy"), np.load("C:/Users/shent/Desktop/Optimal-Individualized-Decision-Making-with-Proxies/data/S1/repeat0/W.npy"), np.load("C:/Users/shent/Desktop/Optimal-Individualized-Decision-Making-with-Proxies/data/S1/repeat0/A.npy"), np.load("C:/Users/shent/Desktop/Optimal-Individualized-Decision-Making-with-Proxies/data/S1/repeat0/Y.npy")
#start = time.time()
#print(ITR_estimation_NN(X,Z,W,A,Y,"C:/Users/shent/Desktop/Optimal-Individualized-Decision-Making-with-Proxies/data/S1/repeat0/h.npy","C:/Users/shent/Desktop/Optimal-Individualized-Decision-Making-with-Proxies/data/S1/repeat0/q1.npy","C:/Users/shent/Desktop/Optimal-Individualized-Decision-Making-with-Proxies/data/S1/repeat0/q0.npy"))
#end = time.time()
#print(end-start)
