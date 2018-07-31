# -*- encoding: utf-8 -*-

import os
from collections import deque
import math
import numpy as np

import matplotlib.pyplot as plt


import tensorflow as tf
# tensorboard --logdir="C:\tensorLogs\xor"

from forwardNet import ForwardNet
from recurrentNet import RecurrentNet
from recurrentResNet import RecurrentResNet
from recResDynNet import RecurrentResDynNet

from mySystem import MySystem, modelPT2

class TestCase(object):
    def __init__(self, timeSteps=2000, stepDuration=0.01, initialX1=0.0, initialX2=0.0, showIndividualPlot=False):
        self.initialX1 = initialX1
        self.initialX2 = initialX2
        self.timeSteps = timeSteps
        self.stepDuration = stepDuration

        self.showIndividualPlot = showIndividualPlot

        self.t = np.linspace(0, int(self.stepDuration * self.timeSteps), timeSteps+1)
        self.u = np.zeros_like(self.t)
        self.netOutput = np.zeros_like(self.t)
        self.y = np.zeros_like(self.t)
        self.yP = np.zeros_like(self.t)
        self.pidOut = np.zeros_like(self.t)
        self.pidControlled = np.zeros_like(self.t)

        self.y[:5] = self.initialX1
        self.yP[:5] = self.initialX2

        self.model = MySystem(modelPT2, initialX1, initialX2)

    def calculateNetAndODE(self, tfSession, net):
        inputU = deque(maxlen=5)
        inputY = deque(maxlen=5)


        for step, _ in enumerate(self.t):

            if step < 5:
                self.y[step] = self.initialX1
                self.yP[step] = 0.0
                inputU.append([self.u[step]])
                inputY.append([self.y[step]])
                continue

            inputU.append([self.u[step]])
            inputY.append([self.y[step]])

            self.netOutput[step] = (tfSession.run(net.y, feed_dict={net.inputU: inputU, net.inputY: inputY}))

            if step < len(self.t) - 1:
                self.y[step + 1], self.yP[step + 1] = self.model.performStep(self.netOutput[step], [self.t[step - 1], self.t[step]])


    def calculateNetAndODERecurrent(self, tfSession, net):

        history_X1 = [[0.0], [0.0]]
        history_X2 = [[0.0], [0.0]]
        history_X3 = [[0.0], [0.0]]
        history_X4 = [[0.0], [0.0]]
        history_X5 = [[0.0], [0.0]]
        history_X6 = [[0.0], [0.0]]

        for step, _ in enumerate(self.t):

            if step < 2:
                self.y[step] = self.initialX1
                self.yP[step] = 0.0
                continue

            input = [[self.u[step]], [self.y[step]]]

            self.netOutput[step], history_X1, history_X2, history_X3, history_X4, history_X5, history_X6 = (tfSession.run(
                [
                    net.y,
                    net.history_01,
                    net.history_02,
                    net.history_03,
                    net.history_04,
                    net.history_05,
                    net.history_06
                ],
                feed_dict={
                    net.input: input,
                    net.history_01: history_X1,
                    net.history_02: history_X2,
                    net.history_03: history_X3,
                    net.history_04: history_X4,
                    net.history_05: history_X5,
                    net.history_06: history_X6
                }))

            if step < len(self.t) - 1:
                self.y[step + 1], self.yP[step + 1] = self.model.performStep(self.netOutput[step], [self.t[step - 1], self.t[step]])

        self.pidOut, self.pidControlled = self.model.getPidControlledResponse(self.t, self.u, self.stepDuration, self.initialX1, self.initialX2)

    def calculateNetAndODERecurrentRes(self, tfSession, net):

        history_X1 = [[0.0], [0.0]]
        history_X2 = [[0.0], [0.0], [0.0], [0.0]]
        history_X3 = history_X1 + history_X2
        history_X4 = history_X2 + history_X3
        history_X5 = history_X3 + history_X4
        history_X6 = history_X4 + history_X5

        for step, _ in enumerate(self.t):

            if step < 2:
                self.y[step] = self.initialX1
                self.yP[step] = 0.0
                continue

            input = [[self.u[step]], [self.y[step]]]

            self.netOutput[step], history_X1, history_X2, history_X3, history_X4, history_X5, history_X6 = (tfSession.run(
                [
                    net.y,
                    net.history_01,
                    net.history_02,
                    net.history_03,
                    net.history_04,
                    net.history_05,
                    net.history_06,
                ],
                feed_dict={
                    net.input: input,
                    net.history_01: history_X1,
                    net.history_02: history_X2,
                    net.history_03: history_X3,
                    net.history_04: history_X4,
                    net.history_05: history_X5,
                    net.history_06: history_X6,
                }))

            if step < len(self.t) - 1:
                self.y[step + 1], self.yP[step + 1] = self.model.performStep(self.netOutput[step], [self.t[step - 1], self.t[step]])

        self.pidOut, self.pidControlled = self.model.getPidControlledResponse(self.t, self.u, self.stepDuration, self.initialX1, self.initialX2)

    def calculateNetAndODERecurrentResDyn(self, tfSession, net):

        history_X1 = [[0.0], [0.0]]
        history_X2 = [[0.0], [0.0], [0.0], [0.0]]
        history_X3 = history_X1 + history_X2
        history_X4 = history_X2 + history_X3
        history_X5 = history_X3 + history_X4
        history_X6 = history_X4 + history_X5

        for step, _ in enumerate(self.t):

            if step < 2:
                self.y[step] = self.initialX1
                self.yP[step] = 0.0
                continue

            input = [[self.u[step]], [self.y[step]]]

            self.netOutput[step], history_X1, history_X2, history_X3, history_X4, history_X5, history_X6 = (tfSession.run(
                [
                    net.y,
                    net.history_01,
                    net.history_02,
                    net.history_03,
                    net.history_04,
                    net.history_05,
                    net.history_06,
                ],
                feed_dict={
                    net.input: input,
                    net.history_01: history_X1,
                    net.history_02: history_X2,
                    net.history_03: history_X3,
                    net.history_04: history_X4,
                    net.history_05: history_X5,
                    net.history_06: history_X6,
                }))

            if step < len(self.t) - 1:
                self.y[step + 1], self.yP[step + 1] = self.model.performStep(self.netOutput[step], [self.t[step - 1], self.t[step]])

        self.pidOut, self.pidControlled = self.model.getPidControlledResponse(self.t, self.u, self.stepDuration, self.initialX1, self.initialX2)


def run(theNetti, sessionFolderOrIndex):

    tf.reset_default_graph()
    # session = tf.Session()

    # ATTENTION: First load the model, then the saver, then start the session -> KEEP THIS ORDER !!
    if theNetti == "control":
        theNet = ForwardNet()

    if theNetti == "recurrent":
        theNet = RecurrentNet()

    if theNetti == "recurrentRes":
        theNet = RecurrentResNet()

    if theNetti == "recurrentResDyn":
        theNet = RecurrentResDynNet(batchSize=1, hiddenLayerCount=6)

    # logDir = "C:\\tensorLogs\\regler"
    logDir = r"D:\00 eigene Daten\000 FH\Ki\tensorCheck\logs"
    runDirs = os.listdir(logDir)


    if type(sessionFolderOrIndex) == int:
        trainDir = os.path.join(logDir, runDirs[sessionFolderOrIndex])
    else:
        trainDir = os.path.join(logDir, sessionFolderOrIndex)


    savePath = os.path.join(trainDir, "savedSession")



    saver = tf.train.Saver()

    session = tf.Session()



    saver.restore(session, savePath)
    print("model loaded from {}".format(savePath))

    # generate case with initial conditions
    testCases = list()
    for i in range(0, 6):
        if i == 0:
            testCases.append(TestCase(showIndividualPlot=True))
        elif i == 1:
            testCases.append(TestCase(showIndividualPlot=True))
        elif i == 2:
            testCases.append(TestCase(showIndividualPlot=True))
        elif i == 3:
            testCases.append(TestCase(initialX1=0.6, showIndividualPlot=True))
        elif i == 4:
            testCases.append(TestCase(initialX1=0.5, showIndividualPlot=True))
        elif i == 5:
            testCases.append(TestCase(initialX1=0.0, showIndividualPlot=True))

    # generate u
    for caseNumber, testCase in enumerate(testCases):
        for i, _ in enumerate(testCase.t):
            if caseNumber == 0:
                if i < 4:
                    testCase.u[i] = 0.0
                elif 4 < i < int(len(testCase.t) / 2):
                    testCase.u[i] = 0.0
                else:
                    testCase.u[i] = 0.0

            elif caseNumber == 1:
                if i < 4:
                    testCase.u[i] = 0.0
                elif 4 < i < int(len(testCase.t) / 2):
                    testCase.u[i] = 1.0
                else:
                    testCase.u[i] = 1.0

            elif caseNumber == 2:
                if i < 4:
                    testCase.u[i] = 0.0
                elif 4 < i < int(len(testCase.t) / 2):
                    testCase.u[i] = 0.3
                else:
                    testCase.u[i] = 0.0

            elif caseNumber == 3:
                if i < 4:
                    testCase.u[i] = 0.0
                elif 4 < i < int(len(testCase.t) / 2):
                    testCase.u[i] = 0.6
                else:
                    testCase.u[i] = 0.0

            elif caseNumber == 4:
                if i < 4:
                    testCase.u[i] = 0.0
                else:
                    testCase.u[i] = 0.5 * math.sin(testCase.t[i]) + 0.5

            elif caseNumber == 5:
                if i < 4:
                    testCase.u[i] = 0.0
                else:
                    testCase.u[i] = 0.05*testCase.t[i]

    if theNetti == "control":
        for testCase in testCases:
            testCase.calculateNetAndODE(session, theNet)

    if theNetti == "recurrent":
        for testCase in testCases:
            testCase.calculateNetAndODERecurrent(session, theNet)

    if theNetti == "recurrentRes" or theNetti == "recurrentResDyn":
        for testCase in testCases:
            testCase.calculateNetAndODERecurrentRes(session, theNet)

    if theNetti == "recurrentResDyn":
        for testCase in testCases:
            testCase.calculateNetAndODERecurrentResDyn(session, theNet)

    # impulseMatrix = session.run(theNet.impulseMatrix)
    # print(impulseMatrix)
    # testSetpoints = list()
    # for i in range(0, len(impulseMatrix)):
    #     testSetpoints.append(1.0)
    # testSetpoints[-1] = theNet.y
    # systemOutput = session.run(theNet.systemOutput, feed_dict={theNet.setPoints: testSetpoints})
    # print(systemOutput)





    session.close()


    sideLength = int(math.sqrt(len(testCases)))
    if sideLength * sideLength < len(testCases):
        sideLength += 1
    rowCount = sideLength
    columnCount = sideLength
    while sideLength * (rowCount - 1) >= len(testCases):
        rowCount -= 1

    figureSubplots, plotAxes = plt.subplots(rowCount, columnCount)

    individualFigures = list()
    for rowNumber, rowAxes in enumerate(plotAxes):
        for columnNumber, axis in enumerate(rowAxes):
            testCaseIndex = rowNumber * len(rowAxes) + columnNumber
            if testCaseIndex < len(testCases):
                testCase = testCases[testCaseIndex]
                axis.plot(testCase.t, testCase.u)
                axis.plot(testCase.t, testCase.netOutput)
                axis.plot(testCase.t, testCase.y)
                axis.set_ylim([-0.5, 1.5])
                if testCase.showIndividualPlot is True:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(testCase.t, testCase.u, "b", label='u', linewidth=1.0)
                    ax.plot(testCase.t, testCase.netOutput, "r", label='netOut', linewidth=0.5)
                    ax.plot(testCase.t, testCase.y, "g", label='netControlled', linewidth=1.0)
                    ax.plot(testCase.t, testCase.pidOut, "r", dashes=[15, 15], label='pidOutput', linewidth=0.5)
                    ax.plot(testCase.t, testCase.pidControlled, "g", dashes=[15, 15], label='pidControlled', linewidth=1.0)
                    ax.legend(loc="best")
                    targetFileName = os.path.join(trainDir, "subplot_{}.png".format(testCaseIndex))
                    fig.savefig(targetFileName, dpi=300, bbox_inches='tight')

    targetFileName = os.path.join(trainDir, "allPlots.png")
    figureSubplots.savefig(targetFileName, dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    # run("control", -1)
    run("recurrentRes", -1)

    # run("control", "run016")
    # run("recurrent", "run018")
