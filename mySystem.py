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



class MySystem(object):
    def __init__(self, modelFunction, x1=0, x2=0, h=0.1, impulseResponseLengthInS=12):
        self.state = [x1, x2]
        self.h = h
        self.impulseResponseLengthInS = impulseResponseLengthInS
        self._tOfImpulseResponse = list()
        self._impulseResponse = list()
        self._stepResponse = list()
        self.modelFunction = modelFunction

    def setState(self, x1, x2):
        self.state = [x1, x2]

    def resetStates(self, newState=list([0, 0])):
        self.state = newState

    # @staticmethod
    # def calculateState(x, t, u):
    #     x1 = x[0]
    #     x2 = x[1]
    #     dx1dt = x2
    #     dx2dt = u - x2 - x1
    #     dxdt = [dx1dt, dx2dt]
    #     return dxdt

    def performStep(self, u, tSpan):
        self.state = odeint(self.modelFunction, self.state, tSpan, args=(u,))[1]
        return self.state

    def getImpulseResponseOverTime(self):

        if len(self._impulseResponse) == 0:

            # calculate a step response of the system
            tAxis = np.linspace(0, self.impulseResponseLengthInS, int(self.impulseResponseLengthInS / self.h) + 1)
            for i in range(0, len(tAxis)):
                self._tOfImpulseResponse.append(tAxis[i])

            oldState = self.state
            self.resetStates()
            self._stepResponse = list()
            self._stepResponse.append([0, 0])
            for i in range(1, len(tAxis)):
                self._stepResponse.append(self.performStep(1.0, [tAxis[i-1], tAxis[i]]))
            self.state = oldState

            # calculate derivative of step response
            self._impulseResponse = list()
            for i in range(0, len(self._stepResponse)):
                if i == 0:
                    self._impulseResponse.append(float(self._stepResponse[1][0] - self._stepResponse[0][0]) / self.h / 2.0)
                elif i == len(self._stepResponse) - 1:
                    self._impulseResponse.append(float(self._stepResponse[i][0] - self._stepResponse[i-1][0]) / self.h / 2.0)
                else:
                    self._impulseResponse.append(float(self._stepResponse[i+1][0] - self._stepResponse[i-1][0]) / 2.0 / self.h)

        return self._tOfImpulseResponse, self._impulseResponse

    @staticmethod
    def getPidControlledResponse(t, u, dt, x1_0, x2_0):
        pidOutList = np.full_like(t, 0.0)
        systemOutList = np.full_like(t, 0.0)
        systemOutList[0] = x1_0

        kPid = 1
        numPid = [-1.0, -1.0]
        denPid = [-0.001, -1.0]
        pid = signal.ZerosPolesGain(numPid, denPid, kPid)
        pidSS = pid.to_ss()
        oldStatePid = [0.0, 0.0]
        # oldStatePid = pidSS.B

        numSys = [1.0]
        denSys = [1.0, 1.0, 1.0]
        testSystem = signal.TransferFunction(numSys, denSys)
        testSystemSS = testSystem.to_ss()
        oldStateSys = [0.0, x1_0]
        # oldStateSys = testSystemSS.B

        for tIndex in range(1, len(t)):
            e = [u[tIndex-1] - systemOutList[tIndex-1], u[tIndex-1] - systemOutList[tIndex-1]]
            tPid, pidOut, oldStatesPid = signal.lsim(pidSS, e, [0.0, dt], X0=oldStatePid)
            oldStatePid = oldStatesPid[-1]
            pidOutList[tIndex] = pidOut[-1]

            # clip the pid to 0.0 - 1.0
            if pidOutList[tIndex] > 1.0:
                pidOutList[tIndex] = 1.0
            if pidOutList[tIndex] < 0.0:
                pidOutList[tIndex] = 0.0

            tSys, sysOut, oldStatesSys = signal.lsim(testSystemSS, [pidOutList[tIndex], pidOutList[tIndex]], [0.0, dt], X0=oldStateSys)
            oldStateSys = oldStatesSys[-1]
            systemOutList[tIndex] = sysOut[-1]

        return pidOutList, systemOutList

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t = np.linspace(0, 10.0, 1000)
    u = np.full_like(t, 1.0)

    pidOut, y = MySystem.getPidControlledResponse(t, u, 0.01, 0.0, 0.0)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    # ax1.set_xlim([0, 6])
    # ax2.set_ylim([-2, 2])
    ax2.plot(t, pidOut, "g--")
    ax2.plot(t, y, "b:")


    plt.show()

