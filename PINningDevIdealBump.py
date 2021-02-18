#!/usr/bin/env python3
"""
ipython
from  PINningDevIdealBump import wrap
wrap(plot=True)
"""

import datetime
import math


import numpy as np
import scipy.io


import matplotlib.pyplot as plt


def wrap(bottled_noise=False, learn=True, run=True, save=False,
         plot=False):
    if plot is True:
        plt.ion()

    if bottled_noise is True:
        loaded = scipy.io.loadmat('shared_randoms.mat')

    g = 1.5
    nRunTot = 125
    nFree = 5
    dtData = 0.0641
    dt = 0.001  # integration step
    tau = 0.01  # 10ms time constant
    P0 = 1
    tauWN = 1
    ampIn = 1
    N = 500
    nLearn = 50
    epochs = 165

    if bottled_noise is True:
        learnList = loaded['learnList'].astype(int) - 1
        learnList = learnList.ravel()
    else:
        learnList = np.random.permutation(N)

    cL = learnList[0:nLearn]
    assert(cL.size == nLearn)
    assert(cL.shape == (nLearn,))
    nCL = learnList[nLearn:]

    tData = np.arange(0, (epochs+1)*dtData, dtData)
    t = np.arange(0, tData[-1], dt)

    xBump = np.zeros((N, len(tData)))

    sig = 0.0343*N  # % scaled correctly in neuron space!!!
    for i in range(N):
        xBump[i, :] = np.exp(-((float(i+1) - N * tData / tData[-1]) ** 2.0) / (2 * sig ** 2))

    hBump = np.log((xBump + 0.01)/(1 - xBump + 0.01))  # current from rate

    ampWN = math.sqrt(tauWN/dt)
    if bottled_noise is True:
        iWN = ampWN*loaded['wn_rand']
    else:
        iWN = ampWN*np.random.randn(N, len(t))
    input = np.ones((N, len(t)))
    for tt in range(1, len(t)):
        input[:, tt] = iWN[:, tt] + (input[:, tt - 1] - iWN[:, tt])*np.exp(- (dt / tauWN))
    input = ampIn*input

    noiseLevel = 0.5
    sigN = noiseLevel * math.sqrt(tau / dt)

    if bottled_noise is True:
        J = g * loaded['weights_rand'] / math.sqrt(N)
    else:
        J = g * np.random.randn(N, N) / math.sqrt(N)
    J0 = J.copy()
    R = np.zeros((N, len(t)))
    JR = np.zeros((N, 1))
#    if save is True:
#        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M")
#        with open('initials_{}.npz'.format(timestamp) , 'wb') as outfile:
#            np.savez(outfile, xBump=xBump, hBump=hBump, J0=J0, input=input)

    if run is True:
        if learn is True:
            PJ = P0 * np.eye(nLearn, nLearn)
        rightTrial = False
        for nRun in range(0, nRunTot):
            print(nRun)
            H = xBump[:, 0, np.newaxis]
            tLearn = 0
            iLearn = 1
            for tt in range(1, len(t)):
                tLearn = tLearn + dt
                R[:, tt, np.newaxis] = 1/(1+np.exp(-H))
                if bottled_noise is True:
                    noise = loaded['stupid_rand']
                else:
                    noise = np.random.randn(N, 1)
                JR = input[:, tt, np.newaxis] + sigN * noise + J.dot(R[:, tt]).reshape((N, 1))
                H = H + dt * (-H + JR) / tau
                if learn is True and tLearn >= dtData and nRun < (nRunTot - nFree):
                    tLearn = 0
                    err = JR[0:N, :] - hBump[0:N, iLearn, np.newaxis]
                    iLearn = iLearn + 1
                    r_slice = R[cL, tt].reshape(nLearn, 1)
                    k = PJ.dot(r_slice)
                    rPr = (r_slice).T.dot(k)[0, 0]
                    c = 1.0/(1.0 + rPr)
                    PJ = PJ - c*(k.dot(k.T))
                    J[0:N, cL.flatten()] = J[:, cL.reshape((nLearn))] - c*np.outer(err.flatten(), k.flatten())
            if plot is True:
                plt.clf()
                plt.imshow(R/R.max())
                plt.colorbar()
                plt.gca().set_aspect(15, adjustable='box')
                plt.draw()
                plt.pause(0.1)
        if save is True:
          timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M")
          with open('finals_{}.npz'.format(timestamp) , 'wb') as outfile:
              np.savez(outfile, J=J, err=err, R=R)

if __name__ == '__main__':
#    wrap(run=True, learn=True, save=True, bottled_noise=True)
    wrap(run=True, learn=True)
