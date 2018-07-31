# -*- encoding: utf-8 -*-

from scipy.integrate import odeint
from scipy import signal
import numpy as np



def modelPT2(x, t, u):
    x1 = x[0]
    x2 = x[1]
    dx1dt = x2
    dx2dt = u - x2 - x1
    dxdt = [dx1dt, dx2dt]
    return dxdt





if __name__ == "__main__":

    import matplotlib.pyplot as plt

    nomSys = [1.0]
    denSys = [1.0, 2.0, 1.0]
    mySys = signal.TransferFunction(nomSys, denSys)

    kSys = 1.0
    nomSys = []
    denSys = [-1.0, -1.0]
    mySys2 = signal.ZerosPolesGain(nomSys, denSys, kSys)



    t, y = signal.step(mySys2)


    figure = plt.figure()
    axis = figure.add_subplot(111)
    # ax1.set_xlim([0, 6])
    # ax2.set_ylim([-2, 2])
    axis.plot(t, y)




    plt.show()

