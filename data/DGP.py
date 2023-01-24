import numpy as np
import random
import math


def DGP(n, scenario_num):

    ######training data generation#######

    #params:
    #n: number of observations generated
    #scenario_num: different number indicate different scenario for evaluation

    p = 2
    b0 = 2
    bX = np.matrix([[0.25],[0.25]])
    t0 = 0.25 

    tX = np.matrix([[0.25],[0.25]])
    alpha0 = 0.25
    alphaA = 0.25
    alphaX = np.matrix([[0.25],[0.25]])
    mu0 = 0.25

    muX =  np.matrix([[0.25],[0.25]])
    kappa0 = 0.25
    kappaA = 0.25
    kappaX = np.matrix([[0.25],[0.25]])
    
    sigmaZW = 0.25
    sigmaZU = 0.5
    sigmaWU = 0.5
    sigmaZ = 1
    sigmaW = 1
    sigmaU = 1
    sigmaY = 0.25
    sigmaX = 0.25
    omega = 2

    theta0 = alpha0-kappa0*sigmaZU/sigmaU**2
    thetaA = alphaA-kappaA*sigmaZU/sigmaU**2
    thetaX = alphaX-kappaX*sigmaZU/sigmaU**2
    thetaU = sigmaZU/sigmaU**2
 
    muA = sigmaWU*kappaA/sigmaU**2 
    tZ = -(kappaA-sigmaWU*muA/sigmaW**2)/thetaU/(sigmaU**2-sigmaWU**2/sigmaW**2)
    tA = -tZ**2*(sigmaZ**2-sigmaZU**2/sigmaU**2)-tZ*thetaA

    sigma1 = np.matrix([[sigmaZ**2, sigmaZW, sigmaZU],[sigmaZW, sigmaW**2,sigmaWU],[sigmaZU, sigmaWU, sigmaU**2]])
   
    ######generation of X, A, Z, W, U###########
    X = np.zeros((n,p))
    for i in range(n):
        for j in range(p):
            X[i,j] = random.normalvariate(mu=muX[j,0], sigma=sigmaX)

    temp1 = t0 + tA + np.dot(X, tX) + tZ *(theta0+thetaA+np.dot(X, thetaX)) +1/2*(1-(sigmaZU**2/sigmaZ**2/sigmaU**2))*tZ**2*sigmaZ**2
    temp2 = tZ*thetaU*(kappa0+kappaA+np.dot(X, kappaX)) + (thetaU*tZ*sigmaU)**2/2
    A = np.zeros((n,1))
    for i in range(n):
        A[i,0] = np.random.binomial(1, 1/(1+math.exp(temp1[i,0]+temp2[i,0])))
    
    
    
    Z = np.zeros((n,1))
    W = np.zeros((n,1))
    U = np.zeros((n,1))
    for i in range(n):
        mu = np.array([alpha0+alphaA*A[i,0]+np.dot(X, alphaX)[i,0], mu0+muA*A[i,0]+np.dot(X, muX)[i,0], kappa0+kappaA*A[i,0]+np.dot(X, kappaX)[i,0]])
        re = np.random.multivariate_normal(mean=mu, cov=sigma1, size=1)
        Z[i,0] = re[0][0]
        W[i,0] = re[0][1]
        U[i,0] = re[0][2]


    ######setup for different scenarios###########
    if scenario_num == 1:
        bA = 0.5 + 3 * X[:,0] - 5 * X[:,1]
        bW = 8
    elif scenario_num == 2:
        bA = 0.5 + 3 * X[:,0] - 5 * X[:,1]
        bW = 8
    elif scenario_num == 3:
        bA = 2.3 + np.abs(X[:,0]-1) - np.abs(X[:,1]+1)
        bW = 4
    elif scenario_num == 4:
        bA = 0.25 - 6 * X[:,0]*X[:,1]
        bW = 5
    elif scenario_num == 5:
        bA = 0.1 - 2*X[:,0]**2
        bW = 8
    else:
        bA = -0.5 + np.exp(X[:,0]) - 3 * X[:,1]
        bW = 8
    
    temp = mu0+muA*A+np.dot(X, muX)+sigmaWU/sigmaU**2*(U-kappa0-kappaA*A-np.dot(X, kappaX))


    ######generation of Y###########
    Y = np.zeros((n,1))
    for i in range(n):
        if scenario_num == 1:
            Y[i,0] = random.normalvariate(mu=b0+bA[i]*A[i,0]+np.dot(X, bX)[i,0]+(bW+0.25*A[i,0]-omega)*temp[i,0]+omega*W[i,0], sigma=sigmaY)
        elif scenario_num == 2:
            Y[i,0] = random.normalvariate(mu=b0+bA[i]*A[i,0]+np.dot(X, bX)[i,0]+(bW-omega)*temp[i,0]+omega*W[i,0], sigma=sigmaY)
        elif scenario_num == 3:
            Y[i,0] = random.normalvariate(mu=b0+bA[i]*A[i,0]+np.sum(X[i,:]**2)+(bW-2.5*A[i,0]+(2*A[i,0]-1)*(np.sin(X[i,0])-2*np.cos(X[i,1]))-omega)*temp[i,0]+omega*W[i,0], sigma=sigmaY)
        elif scenario_num == 4:
            Y[i,0] = random.normalvariate(mu=b0+bA[i]*A[i,0]+np.sum(X[i,:]**2)+(bW-omega)*temp[i,0]+omega*W[i,0], sigma=sigmaY)
        elif scenario_num == 5:
            Y[i,0] = random.normalvariate(mu=b0+bA[i]*A[i,0]+np.sum(X[i,:]**2)+(bW+0.8*A[i,0]+(2*A[i,0]-1)*(4*X[i,1]**2)-omega)*temp[i,0]+omega*W[i,0], sigma=sigmaY)
        else:
            Y[i,0] = random.normalvariate(mu=b0+bA[i]*A[i,0]+np.dot(X, bX)[i,0]+(bW-omega)*temp[i,0]+omega*W[i,0], sigma=sigmaY)

    
    A = 2*A -1

    return X, Z, W, U, A, Y

def DGP_test(n):
    
    ######test data generation#######

    #params:
    #n: number of test points generated

    ######fixed parameter setup#######
    p = 2
   
    alpha0 = 0.25
    alphaA = 0.25
    alphaX = np.matrix([[0.25],[0.25]])
    mu0 = 0.25
    muA = 0.125
    muX = np.matrix([[0.25],[0.25]])
    kappa0 = 0.25
    kappaA = 0.25
    kappaX = np.matrix([[0.25],[0.25]])

    
    sigmaZW = 0.25
    sigmaZU = 0.5
    sigmaWU = 0.5
    sigmaZ = 1
    sigmaW = 1
    sigmaU = 1
    sigmaX =  0.25

    theta0 = alpha0-kappa0*sigmaZU/sigmaU**2
    thetaA = alphaA-kappaA*sigmaZU/sigmaU**2
    thetaX = alphaX-kappaX*sigmaZU/sigmaU**2
    thetaU = sigmaZU/sigmaU**2
    t0 = 0.25 
    tZ = -(kappaA-sigmaWU*muA/sigmaW**2)/thetaU/(sigmaU**2-sigmaWU**2/sigmaW**2)
    tA = -tZ**2*(sigmaZ**2-sigmaZU**2/sigmaU**2)-tZ*thetaA
    tX = np.matrix([[0.25],[0.25]])
    
    #np.matrix([[0.125],[0.125]])
    sigma1 = np.matrix([[sigmaZ**2, sigmaZW, sigmaZU],[sigmaZW, sigmaW**2,sigmaWU],[sigmaZU, sigmaWU, sigmaU**2]])


    ######Generation of X,Z,W,U#######
    X = np.zeros((n,p))
    for i in range(n):
        for j in range(p):
            X[i,j] = random.normalvariate(mu=muX[j,0], sigma=sigmaX)
    theta = 1/(1+ np.exp(t0+tA+np.dot(X, tX)+tZ*(theta0+thetaA+np.dot(X,thetaX))+tZ**2*sigmaZ**2*(1-sigmaZU**2/(sigmaZ**2*sigmaU**2))/2+tZ*thetaU*(kappa0+kappaA+np.dot(X,kappaX))+sigmaU**2*tZ**2*thetaU**2/2))

    Z = np.zeros((n,1))
    W = np.zeros((n,1))
    U = np.zeros((n,1))
    for i in range(n):
        mu = np.array([alpha0+alphaA*theta[i,0]+np.dot(X, alphaX)[i,0], mu0+muA*theta[i,0]+np.dot(X, muX)[i,0], kappa0+kappaA*theta[i,0]+np.dot(X, kappaX)[i,0]])
        re = np.random.multivariate_normal(mean=mu, cov=sigma1, size=1)
        Z[i,0] = re[0][0]
        W[i,0] = re[0][1]
        U[i,0] = re[0][2]
   
    return X, Z, W, U
    
def DGP_eval(n, scenario_num, A2, X, W, Z, U):

    ######evaluation for test data#######

    #params:
    #n: number of test points generated
    #scenario_num: scenario selected
    #A2: treatment assigned (-1 or 1)
    #X,W,Z,U: resulting covariates from DGP_test(n)

    A = (A2+1)/2
        
    b0 = 2
    bX = np.matrix([[0.25],[0.25]])
    omega = 2
    mu0 = 0.25
    muX = np.matrix([[0.25],[0.25]])
    kappa0 = 0.25
    kappaA = 0.25
    kappaX = np.matrix([[0.25],[0.25]])
    
    sigmaWU = 0.5
    sigmaY = 0.25
    sigmaU = 1
    muA = sigmaWU*kappaA/sigmaU**2 
    
    ######setup for different scenarios###########
    if scenario_num == 1:
        bA = 0.5 + 3 * X[:,0] - 5 * X[:,1]
        bW = 8
    elif scenario_num == 2:
        bA = 0.5 + 3 * X[:,0] - 5 * X[:,1]
        bW = 8
    elif scenario_num == 3:
        bA = 2.3 + np.abs(X[:,0]-1) - np.abs(X[:,1]+1)
        bW = 4
    elif scenario_num == 4:
        bA = 0.25 - 6 * X[:,0]*X[:,1]
        bW = 5
    elif scenario_num == 5:
        bA = 0.1 - 2*X[:,0]**2
        bW = 8
    else:
        bA = -0.5 + np.exp(X[:,0]) - 3 * X[:,1]
        bW = 8
    
    temp = mu0+muA*A+np.dot(X, muX)+sigmaWU/sigmaU**2*(U-kappa0-kappaA*A-np.dot(X, kappaX))

    
    ######generation of Y###########
    Y = np.zeros((n,1))
    for i in range(n):
        if scenario_num == 1:
            Y[i,0] = random.normalvariate(mu=b0+bA[i]*A[i,0]+np.dot(X, bX)[i,0]+(bW+0.25*A[i,0]-omega)*temp[i,0]+omega*W[i,0], sigma=sigmaY)
        elif scenario_num == 2:
            Y[i,0] = random.normalvariate(mu=b0+bA[i]*A[i,0]+np.dot(X, bX)[i,0]+(bW-omega)*temp[i,0]+omega*W[i,0], sigma=sigmaY)
        elif scenario_num == 3:
            Y[i,0] = random.normalvariate(mu=b0+bA[i]*A[i,0]+np.sum(X[i,:]**2)+(bW-2.5*A[i,0]+(2*A[i,0]-1)*(np.sin(X[i,0])-2*np.cos(X[i,1]))-omega)*temp[i,0]+omega*W[i,0], sigma=sigmaY)
        elif scenario_num == 4:
            Y[i,0] = random.normalvariate(mu=b0+bA[i]*A[i,0]+np.sum(X[i,:]**2)+(bW-omega)*temp[i,0]+omega*W[i,0], sigma=sigmaY)
        elif scenario_num == 5:
            Y[i,0] = random.normalvariate(mu=b0+bA[i]*A[i,0]+np.sum(X[i,:]**2)+(bW+0.8*A[i,0]+(2*A[i,0]-1)*(4*X[i,1]**2)-omega)*temp[i,0]+omega*W[i,0], sigma=sigmaY)
        else:
            Y[i,0] = random.normalvariate(mu=b0+bA[i]*A[i,0]+np.dot(X, bX)[i,0]+(bW-omega)*temp[i,0]+omega*W[i,0], sigma=sigmaY)

    return Y
  