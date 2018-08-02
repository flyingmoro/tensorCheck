# -*- encoding: utf-8 -*-


# tensorboard --logdir="C:\tensorLogs\regler"
# tensorboard --logdir="C:\Users\H5489\Documents\00 BA\tensorLogs"
from collections import deque

from random import shuffle
import os, shutil

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import tensorflow as tf

from mySystem import MySystem, modelPT2
from forwardNet import ForwardNet
from trainingDataGenerator import TrainingDataGenerator, TrainingDataParameters, TrainingData
import evaluateTrainedNet


class ControllerTrainer(object):
    def __init__(self, trainingData):
        self.trainingData = trainingData
        self.odeStates = [0, 0]
        self.net = ForwardNet()
        self.trainDir = ""
        self.session = None
        self.saver = None
        self.train_writer = None

    def train(self, learningRate=0.1, epochs=1):

        tf.summary.histogram('WU1', self.net.inputWU1)
        tf.summary.histogram('WY1', self.net.inputWY1)
        tf.summary.histogram('W1', self.net.W1)
        tf.summary.histogram('W2', self.net.W2)
        tf.summary.histogram('W3', self.net.W3)
        tf.summary.histogram('W4', self.net.W4)
        tf.summary.histogram('W5', self.net.W5)
        tf.summary.scalar('trainingCost', self.net.trainingCost)
        myOwnSummary = tf.Summary()

        mergedSummaries = tf.summary.merge_all()

        # train_step = tf.train.AdagradOptimizer(learningRate).minimize(self.net.cost)
        train_step = tf.train.AdagradOptimizer(learningRate).minimize(self.net.trainingCost)

        # logDir = "C:\\tensorLogs\\regler"
        logDir = r"C:\Users\H5489\Documents\00 BA\tensorLogs"
        if not os.path.isdir(logDir):
            os.makedirs(logDir)
        dirs = os.listdir(logDir)
        runString = str(len(dirs) + 1).zfill(3)
        self.trainDir = os.path.join(logDir, "run{}".format(runString))

        self.saver = tf.train.Saver()


        self.session = tf.Session()
        self.train_writer = tf.summary.FileWriter(self.trainDir, graph=self.session.graph)
        tf.global_variables_initializer().run(session=self.session)


        print("len trainingData", len(self.trainingData))



        expectedTrainingRounds = epochs * len(self.trainingData) * len(self.trainingData[0].tValues)
        print("expecting {} training rounds".format(expectedTrainingRounds))
        meanErrorPerBatch = list()


        trainingCounter = 0
        for epoch in range(1, epochs + 1):

            # select training data
            for stepResponseCounter, stepResponse in enumerate(self.trainingData):

                errorPerStepResponseSum = 0
                uIn = deque(maxlen=5)
                yIn = deque(maxlen=5)

                for timePointIndex, _ in enumerate(stepResponse.shortT):

                    # select u and precalculated y as input and targetU as target
                    # if timePointIndex < 5:
                    #     continue
                    # input = list()
                    # target = [[stepResponse.targetSetPoints[timePointIndex]]]
                    # for d in range(0, 10):
                    #     input.append(list())
                    # for k in range(0, 5):
                    #     input[4 - k].append(stepResponse.uValues[timePointIndex - k])
                    #     input[5 + 4 - k].append(stepResponse.yValues[timePointIndex - k])
                    #
                    # # perform training
                    #  _, specialCost, trainingCost, lastSummary = self.session.run(
                    #              [train_step, self.net.cost, self.net.trainingCost, mergedSummaries],
                    #              feed_dict={self.net.inputLayer: input, self.net.t: target})



                    # select u and precalculated y as input and targetY as target
                    # l = len(stepResponse.uValues)
                    uIn.append([stepResponse.shortU[timePointIndex]])
                    yIn.append([stepResponse.shortY[timePointIndex]])

                    # skip first 5 steps
                    if timePointIndex < 5:
                        continue

                    target = [[stepResponse.shortTargets[timePointIndex]]]

                    # perform training
                    if timePointIndex < len(stepResponse.shortT) - 2:
                        _, specialCost, trainingCost = self.session.run(
                                     [train_step, self.net.cost, self.net.trainingCost],
                                     feed_dict={self.net.inputU: uIn, self.net.inputY: yIn, self.net.t: target})

                    else:
                        _, specialCost, trainingCost, lastSummary = self.session.run(
                                     [train_step, self.net.cost, self.net.trainingCost, mergedSummaries],
                                     feed_dict={self.net.inputU: uIn, self.net.inputY: yIn, self.net.t: target})
                        self.train_writer.add_summary(lastSummary, trainingCounter)

                    errorPerStepResponseSum += specialCost[0][0]
                    trainingCounter += 1

                # calculate some statistics and output progress
                lastMeanError = errorPerStepResponseSum / float(len(stepResponse.tValues))
                meanErrorPerBatch.append(lastMeanError)
                print("last mean error: {:.12f}, round {}, remaining rounds: {}".format(
                    float(meanErrorPerBatch[stepResponseCounter]), trainingCounter, expectedTrainingRounds - trainingCounter))
                myOwnSummary.value.add(tag="MeanErrorPerTrainingCurve", simple_value=lastMeanError)
                self.train_writer.add_summary(myOwnSummary, trainingCounter)


        self.abortLearning()
        return meanErrorPerBatch, self.trainDir

    def abortLearning(self):
        savePath = os.path.join(self.trainDir, "savedSession")
        self.saver.save(self.session, savePath)

        self.train_writer.close()
        self.session.close()


















if __name__ == "__main__":
    labelParamsList = list()
    for initY in range(0, 1): # 0, 11
    # for initY in range(1, 11): # 0, 11
        initialY = float(initY) * 0.1
        for i in range(3, 5): # 0, 30
        # for i in range(0, 10): # 0, 30
            timeOfUChange = float(i) * 0.1
            uHighLevel = 1
            uLowLevel = 0

            # rising energy
            labelParams = TrainingDataParameters()
            labelParams.initialY = initialY
            labelParams.timeOfUChange = timeOfUChange
            labelParams.uLevelBegin = uHighLevel
            labelParams.uLevelEnd = uLowLevel
            labelParams.zeroFillCount = 100
            labelParams.odeModel = MySystem(modelPT2, x1=initialY)
            labelParamsList.append(labelParams)

            # falling energy
            labelParams1 = TrainingDataParameters()
            labelParams1.initialY = initialY
            labelParams1.timeOfUChange = timeOfUChange
            labelParams1.uLevelBegin = uLowLevel
            labelParams1.uLevelEnd = uHighLevel
            labelParams1.zeroFillCount = 100
            labelParams1.odeModel = MySystem(modelPT2, x1=initialY)
            labelParamsList.append(labelParams1)


    labels = list()
    for labelParams in labelParamsList:
        labels.append(TrainingDataGenerator.createData(labelParams))



    extractedLabels = list()
    for i, label in enumerate(labels):

        # if label.yValues[0] != 0.5:
        #     continue
        if label.excessiveY is True:
            continue
        if label.noDirectionChange is True:
            continue



        extractedLabels.append(label)


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim([0, 6])
    # ax.set_ylim([0, len(labels)])
    # ax.set_zlim([0, 1])

    # for i, label in enumerate(labels):
    #     # if i == 21:
    #     # ax.plot(label.tValues, label.uValues, i, zdir='y')
    #     ax.plot(label.tValues, label.yValues, i, zdir='y')
    #     # ax.plot(label.tValues, label.yPValues, i, zdir='y')
    #     # ax.plot(label.tValues, label.uForLabeling, i, zdir='y')
    #     ax.plot(label.tValues, label.uValues, i, zdir='y')



    print("starting to learn")
    shuffle(extractedLabels)
    theTrainer = ControllerTrainer(extractedLabels)
    try:
        meanErrorPerBatch, trainDir = theTrainer.train(learningRate=0.1, epochs=15)
    except (KeyboardInterrupt, SystemExit):
        print("stopping...")
        theTrainer.abortLearning()
        raise


    del theTrainer

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    # ax1.set_xlim([0, 6])
    # ax1.set_ylim([0, len(labels)])

    xValues = range(0, len(meanErrorPerBatch))
    ax1.plot(xValues, meanErrorPerBatch)



    evaluateTrainedNet.run("control", -1)

    # save the model
    shutil.copyfile("recurrentNet.py", os.path.join(trainDir, "recurrentNet.py"))

    plt.show()
