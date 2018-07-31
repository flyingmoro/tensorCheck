# -*- encoding: utf-8 -*-



def numericDerivative(y, dt=1.0):
    derivative = list()
    for i in range(0, len(y)):
        if i == 0:
            derivative.append(float(y[1] - y[0]) / dt)
        elif i == len(y) - 1:
            derivative.append(float(y[i] - y[i-1]) / dt)
        else:
            derivative.append(float(y[i+1] - y[i-1]) / dt / 2.0)
    return derivative


def getUFromY(y, dt=1.0):
    yp = numericDerivative(y, dt)
    ypp = numericDerivative(yp, dt)
    us = list()
    for i in range(len(y)):
        us.append(y[i] + yp[i] + ypp[i])
    return us