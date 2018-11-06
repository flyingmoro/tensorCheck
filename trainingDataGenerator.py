# -*- encoding: utf-8 -*-

from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mySystem import MySystem, modelPT2
import helpers


class TrainingDataParameters(object):
    def __init__(self):
        self.initialY = 0.0
        self.timeOfUChange = 0.5
        self.uLevelBegin = 1.0
        self.uLevelEnd = 0.0
        self.odeDuration = 5
        self.timeStepCount = 501
        self.zeroFillCount = 0
        self.odeModel = None



class TrainingData(object):
    def __init__(self):
        self.tValues = list()
        self.uValues = list()
        self.yValues = list()
        self.yPValues = list()
        self.targetSetPoints = list()
        self.shortT = list()
        self.shortU = list()
        self.shortY = list()
        self.shortTargets = list()
        self.uForLabeling = list()
        self.timeOfDerivativeEqualsZero = 0
        self.yOfDerivativeEqualsZero = 0
        self.excessiveY = False
        self.noDirectionChange = False



class MyHelper(object):
    @staticmethod
    def getIndexOfMinimum(oneDArray):
        min = 10000000000
        lastIndex = -1
        for i, value in enumerate(oneDArray):
            if value < min:
                min = value
                lastIndex = i

        return lastIndex

    @staticmethod
    def getIndexOfMaximum(oneDArray):
        max = -10000000000
        lastIndex = -1
        for i, value in enumerate(oneDArray):
            if value > max:
                max = value
                lastIndex = i

        return lastIndex







class TrainingDataGenerator(object):
    def __init__(self):
        pass

    @staticmethod
    def calculateUForGivenY(y, impulseResponse, h=1000):

        u = list()
        for i in range(0, len(impulseResponse)):
            u.append(y[0])
        for step in range(0, len(y) - 1):
            nomSum = float(y[step+1]) / h
            for impulseIndex in range(1, len(impulseResponse)):
                uIndex = (impulseIndex + 1) * -1
                nomSum -= impulseResponse[impulseIndex] * u[uIndex]
                # print(impulseIndex, uIndex)
            newU = nomSum / impulseResponse[0]
            # newU = nomSum / impulseResponse[0] / 35
            # newU = nomSum
            # u.append(nomSum / impulseResponse[0])
            # u.append(nomSum / impulseResponse[0] / 35)
            # if newU > 1:
            #     u.append(1.0)
            # elif newU < 0.0:
            #     u.append(0.0)
            # else:
            #     u.append(newU)
            u.append(newU)

        for i in range(0, len(impulseResponse) - 1):
            u.pop(0)

        return u


    @staticmethod
    def strangeStuff(y, dt=0.1):

        u = helpers.getUFromY(y, dt)

        return u


    @staticmethod
    def createData(labelParameters):


        t = np.linspace(0, labelParameters.odeDuration, labelParameters.timeStepCount)

        # generate a step like function
        u = np.full_like(t, labelParameters.uLevelBegin)
        u[int(labelParameters.timeOfUChange * 100) + 1:] = labelParameters.uLevelEnd

        # calculate ode
        y = np.empty_like(t)
        yP = np.empty_like(t)
        y[0] = labelParameters.initialY
        yP[0] = 0.0
        for step in range(1, labelParameters.timeStepCount):
            y[step], yP[step] = labelParameters.odeModel.performStep(u[step], [t[step - 1], t[step]])





        excessiveY = False
        if labelParameters.uLevelBegin > labelParameters.uLevelEnd:
            minOrMax = "maximum"
            indexOfYP0 = MyHelper.getIndexOfMaximum(y)
            tOfYP0 = t[indexOfYP0]
            yP0 = y[indexOfYP0]
            if yP0 > labelParameters.uLevelBegin:
                excessiveY = True
        else:
            minOrMax = "minimum"
            indexOfYP0 = MyHelper.getIndexOfMinimum(y)
            tOfYP0 = t[indexOfYP0]
            yP0 = y[indexOfYP0]
            if yP0 < labelParameters.uLevelBegin:
                excessiveY = True

        noDirectionChange = False
        if indexOfYP0 == len(y) - 1:
            noDirectionChange = True



        u[indexOfYP0:] = yP0
        y[indexOfYP0 + 1:len(y)] = yP0
        uForLabeling = np.full_like(t, yP0)



        shortT = t[0:indexOfYP0]
        shortU = uForLabeling[0:indexOfYP0]
        shortY = y[0:indexOfYP0]
        shortTargets = u[0:indexOfYP0]




        # make all training samples the same length
        timeSpan = t[1] - t[0]
        while len(y) < labelParameters.zeroFillCount + labelParameters.timeStepCount:
            t = np.insert(t, 0, t[0] - timeSpan)
            u = np.insert(u, 0, labelParameters.initialY)
            uForLabeling = np.insert(uForLabeling, 0, labelParameters.initialY)
            y = np.insert(y, 0, labelParameters.initialY)
            yP = np.insert(yP, 0, 0)
        if t[0] < 0:
            offset = t[0]
            for i, element in enumerate(t):
                t[i] -= offset


        # _, impulseResponse = labelParameters.odeModel.getImpulseResponseOverTime()
        # convolutionU = TrainingDataGenerator.calculateUForGivenY(y, impulseResponse, h=0.1)

        # strangeStuff = TrainingDataGenerator.strangeStuff(y)

        trainingData = TrainingData()
        trainingData.tValues = t
        trainingData.uValues = uForLabeling
        trainingData.yValues = y
        trainingData.yPValues = yP
        trainingData.targetSetPoints = u

        trainingData.shortT = shortT
        trainingData.shortU = shortU
        trainingData.shortY = shortY
        trainingData.shortTargets = shortTargets

        trainingData.timeOfDerivativeEqualsZero = tOfYP0
        trainingData.yOfDerivativeEqualsZero = yP0
        trainingData.excessiveY = excessiveY
        trainingData.noDirectionChange = noDirectionChange


        return trainingData





if __name__ == "__main__":


    trainingParameterSets = list()
    for initY in range(9, 10): # 0, 11
        initialY = float(initY) * 0.1
        for i in range(3, 4): # 0, 30
            timeOfUChange = float(i) * 0.1
            uHighLevel = 1
            uLowLevel = 0

            # rising energy
            paramsHighToLow = TrainingDataParameters()
            paramsHighToLow.initialY = initialY
            paramsHighToLow.timeOfUChange = timeOfUChange
            paramsHighToLow.uLevelBegin = uHighLevel
            paramsHighToLow.uLevelEnd = uLowLevel
            paramsHighToLow.zeroFillCount = 100
            paramsHighToLow.odeModel = MySystem(modelPT2, x1=initialY)
            trainingParameterSets.append(paramsHighToLow)

            # falling energy
            paramsLowToHigh = TrainingDataParameters()
            paramsLowToHigh.initialY = initialY
            paramsLowToHigh.timeOfUChange = timeOfUChange
            paramsLowToHigh.uLevelBegin = uLowLevel
            paramsLowToHigh.uLevelEnd = uHighLevel
            paramsLowToHigh.zeroFillCount = 100
            paramsLowToHigh.odeModel = MySystem(modelPT2, x1=initialY)
            trainingParameterSets.append(paramsLowToHigh)


    trainingData = list()
    for trainingParameterSet in trainingParameterSets:
        trainingData.append(TrainingDataGenerator.createData(trainingParameterSet))



    extractedTrainingData = list()
    for i, label in enumerate(trainingData):
        if label.excessiveY is True:
            continue
        if label.noDirectionChange is True:
            continue
        extractedTrainingData.append(label)



    # count


    # shuffle(extractedTrainingData)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.set_xlim([0, 6])
    # ax.set_ylim([0, len(extractedTrainingData)])
    # ax.set_zlim([-10, 10])
    # for i, tData in enumerate(extractedTrainingData):
    #     # ax.plot(tData.tValues, tData.uValues, i, zdir='y')
    #     ax.plot(tData.tValues, tData.targetSetPoints, i, zdir='y')
    #     ax.plot(tData.tValues, tData.targetSetPointsConvolution, i, zdir='y')
    #     # ax.plot(tData.tValues, tData.yValues, i, zdir='y')
    #     # ax.plot(tData.tValues, tData.yPValues, i, zdir='y')


    tDataIndex = 1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_xlim([0, 6])
    # ax.set_ylim([-1.1, 1.1])
    ax.plot(extractedTrainingData[tDataIndex].tValues, extractedTrainingData[tDataIndex].uValues)
    ax.plot(extractedTrainingData[tDataIndex].tValues, extractedTrainingData[tDataIndex].yValues)
    ax.plot(extractedTrainingData[tDataIndex].tValues, extractedTrainingData[tDataIndex].targetSetPoints)



    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    # ax.set_xlim([0, 6])
    # ax.set_ylim([-1.1, 1.1])
    ax1.plot(extractedTrainingData[tDataIndex].shortT, extractedTrainingData[tDataIndex].shortU)
    ax1.plot(extractedTrainingData[tDataIndex].shortT, extractedTrainingData[tDataIndex].shortY)
    ax1.plot(extractedTrainingData[tDataIndex].shortT, extractedTrainingData[tDataIndex].shortTargets)



    plt.show()
