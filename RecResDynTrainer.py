# -*- encoding: utf-8 -*-


# tensorboard --logdir="C:\tensorLogs\regler"
from collections import deque

from random import shuffle
import os, shutil

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import tensorflow as tf

from mySystem import MySystem, modelPT2
from recResDynNet import RecurrentResDynNet
from trainingDataGenerator import TrainingDataGenerator, TrainingDataParameters, TrainingData
import evaluateTrainedNet


class RecurrentResDynTrainer(object):
    def __init__(self, trainingData, batchSize=2, hiddenLayerCount=5):
        self.trainingData = trainingData
        self.odeStates = [0, 0]
        self.net = RecurrentResDynNet(batchSize=batchSize, hiddenLayerCount=hiddenLayerCount)
        self.trainDir = ""
        self.session = None
        self.saver = None
        self.train_writer = None

    def train(self, learningRate=0.1, epochCount=1):
        myOwnSummary = tf.Summary()
        self.mergedSummaries = tf.summary.merge_all()

        grads = None

        # logDir = "C:\\tensorLogs\\regler"
        logDir = r"D:\00 eigene Daten\000 FH\S 7 BA\tensorCheck\logs"
        if not os.path.isdir(logDir):
            os.makedirs(logDir)
        dirs = os.listdir(logDir)
        runString = str(len(dirs) + 1).zfill(3)
        self.trainDir = os.path.join(logDir, "run{}".format(runString))

        self.saver = tf.train.Saver()


        self.session = tf.Session()
        self.train_writer = tf.summary.FileWriter(self.trainDir, graph=self.session.graph)
        tf.global_variables_initializer().run(session=self.session)


        batchCountPerEpoch = int(len(self.trainingData) / self.net.batchSize)
        totalBatchCount = batchCountPerEpoch * epochCount
        stepResponseSampleCount = len(self.trainingData[0].tValues)
        trainingIterationCount = epochCount * batchCountPerEpoch * stepResponseSampleCount

        print("{} epochs, {} batches, {} training iterations to do".format(epochCount, totalBatchCount, trainingIterationCount))
        meanErrorPerBatch = list()


        trainingIterationsCounter = 0
        totalBatchesCounter = 0
        for currentEpoch in range(epochCount):

            errorPerStepResponseSum = 0
            uIn = deque(maxlen=1)
            yIn = deque(maxlen=1)

            history_X1 = [[0.0] * self.net.batchSize] * 2
            history_X2 = [[0.0] * self.net.batchSize] * 4
            history_X3 = history_X1 + history_X2
            history_X4 = history_X2 + history_X3
            history_X5 = history_X3 + history_X4
            history_X6 = history_X4 + history_X5

            # first iterate over batchesCount, for the history activations to update properly
            for currentBatchIndex in range(batchCountPerEpoch):
                for timePointIndex in range(stepResponseSampleCount):

                    # select u and precalculated y as input and targetY as target
                    # create batches to run batchSize stepResponses in parallel
                    input = list()
                    target = list()
                    # input.append([stepResponse.shortU[timePointIndex]])
                    # input.append([stepResponse.shortY[timePointIndex]])
                    # target = [[stepResponse.shortTargets[timePointIndex]]]
                    inputBatchesU = list()
                    inputBatchesY = list()
                    targetBatches = list()

                    for stepResponseOffset in range(self.net.batchSize):
                        inputBatchesU.append(self.trainingData[currentBatchIndex + stepResponseOffset].uValues[timePointIndex])
                        inputBatchesY.append(self.trainingData[currentBatchIndex + stepResponseOffset].yValues[timePointIndex])

                        targetBatches.append(self.trainingData[currentBatchIndex + stepResponseOffset].targetSetPoints[timePointIndex])

                    input.append(inputBatchesU)
                    input.append(inputBatchesY)
                    target.append(targetBatches)

                    # perform training
                    if timePointIndex < stepResponseSampleCount - 2:
                        _, specialCost, trainingCost, history_X1, history_X2, history_X3, history_X4, history_X5, history_X6, grads = self.session.run([
                                         self.net.train_step,
                                         self.net.cost,
                                         self.net.trainingCost,
                                         self.net.allLayerOps[1]["history"],
                                         self.net.allLayerOps[2]["history"],
                                         self.net.allLayerOps[3]["history"],
                                         self.net.allLayerOps[4]["history"],
                                         self.net.allLayerOps[5]["history"],
                                         self.net.allLayerOps[6]["history"],
                                         self.net.gradients
                        ],
                                     feed_dict={
                                         self.net.allLayerOps[0]["output"]: input,
                                         self.net.allLayerOps[1]["history"]: history_X1,
                                         self.net.allLayerOps[2]["history"]: history_X2,
                                         self.net.allLayerOps[3]["history"]: history_X3,
                                         self.net.allLayerOps[4]["history"]: history_X4,
                                         self.net.allLayerOps[5]["history"]: history_X5,
                                         self.net.allLayerOps[6]["history"]: history_X6,
                                         self.net.t: target
                                     })
                        # print(trainingCost)
                        # for subGrads in grads:
                        #     print("grads:")
                        #     print(subGrads[0])
                        #     print("weights:")
                        #     print(subGrads[1])
                        # print(" -------------------------------------- ")
                    else:
                        _, specialCost, trainingCost, history_X1, history_X2, history_X3, history_X4, history_X5, history_X6, lastSummary = self.session.run([
                                         self.net.train_step,
                                         self.net.cost,
                                         self.net.trainingCost,
                                         self.net.allLayerOps[1]["history"],
                                         self.net.allLayerOps[2]["history"],
                                         self.net.allLayerOps[3]["history"],
                                         self.net.allLayerOps[4]["history"],
                                         self.net.allLayerOps[5]["history"],
                                         self.net.allLayerOps[6]["history"],
                                         self.mergedSummaries
                                     ],
                                     feed_dict={
                                         self.net.allLayerOps[0]["output"]: input,
                                         self.net.allLayerOps[1]["history"]: history_X1,
                                         self.net.allLayerOps[2]["history"]: history_X2,
                                         self.net.allLayerOps[3]["history"]: history_X3,
                                         self.net.allLayerOps[4]["history"]: history_X4,
                                         self.net.allLayerOps[5]["history"]: history_X5,
                                         self.net.allLayerOps[6]["history"]: history_X6,
                                         self.net.t: target
                                     })
                        self.train_writer.add_summary(lastSummary, trainingIterationsCounter)

                    errorPerStepResponseSum += specialCost[0][0]
                    trainingIterationsCounter += 1

                # calculate some statistics and output progress
                lastMeanError = errorPerStepResponseSum / float(stepResponseSampleCount)
                meanErrorPerBatch.append(lastMeanError)
                remainingEpochs = epochCount - currentEpoch
                remainingBatches =  totalBatchCount - totalBatchesCounter
                remainingTrainingIterations = trainingIterationCount - trainingIterationsCounter
                print("last mean error: {:.12f}, iteration {}, remaining: {}, batch {}, remaining: {}, epoch {}, remaining: {}".format(
                    float(meanErrorPerBatch[-1]),
                    trainingIterationsCounter, remainingTrainingIterations,
                    totalBatchesCounter, remainingBatches,
                    currentEpoch, remainingEpochs))
                myOwnSummary.value.add(tag="MeanErrorPerTrainingCurve", simple_value=lastMeanError)
                self.train_writer.add_summary(myOwnSummary, trainingIterationsCounter)

                totalBatchesCounter += 1


        self.abortLearning()
        return meanErrorPerBatch, self.trainDir

    def abortLearning(self):
        savePath = os.path.join(self.trainDir, "savedSession")
        self.saver.save(self.session, savePath)

        self.train_writer.close()
        self.session.close()


















if __name__ == "__main__":
    labelParamsList = list()
    for initY in range(7, 8): # 0, 11
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

    shuffle(extractedLabels)


    print("starting to learn, stepResponses count: {}".format(len(extractedLabels)))
    theTrainer = RecurrentResDynTrainer(extractedLabels, batchSize=1, hiddenLayerCount=6)
    try:
        meanErrorPerBatch, trainDir = theTrainer.train(learningRate=0.5, epochCount=3)
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



    evaluateTrainedNet.run("recurrentResDyn", -1)


    # save the model
    shutil.copyfile("recurrentNet.py", os.path.join(trainDir, "recurrentNet.py"))

    plt.show()

