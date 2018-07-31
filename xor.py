# -*- encoding: utf-8 -*-
import os

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import tensorflow as tf
# tensorboard --logdir="C:\tensorLogs\xor"



def run():

    # ++++++++++ epoch 1 ++++++++++++++
    # y [array([[-1.,  0.]], dtype=float32), array([[ 0.22320993]], dtype=float32)]
    # W1  [[ 0.653404   -0.11311577]
    #  [-0.67821217 -0.66344476]]
    # B1  [ 0.         -0.11403587]
    # W2  [[-0.77193332 -0.30499983]
    #  [ 0.55443454  0.93072635]]
    # B2  [-0.07312778 -0.07623759]
    # W3  [[-0.2986654 ]
    #  [-0.29348287]]
    # B3  [ 0.22320993]
    # cost  0.301701
    # ++++++++++ epoch 3000 ++++++++++++++
    # y [array([[ 1.,  1.]], dtype=float32), array([[ 0.]], dtype=float32)]
    # W1  [[ 1.3795079   0.98846519]
    #  [-1.36440146 -0.98846519]]
    # B1  [-0.07842212  0.98846465]
    # W2  [[-1.12615657 -1.1106869 ]
    #  [ 0.73257029  0.72127217]]
    # B2  [ -2.56500098e-05   3.86347097e-07]
    # W3  [[-0.69880587]
    #  [-0.69289184]]
    # B3  [ 1.00000024]
    # cost  0.0
    # [array([[ 0.,  0.]], dtype=float32), array([[ 0.]], dtype=float32), 0.0]
    # [array([[ 0.,  1.]], dtype=float32), array([[ 1.]], dtype=float32), 0.0]
    # [array([[ 1.,  0.]], dtype=float32), array([[ 1.00000024]], dtype=float32), 2.8421709e-14]
    # [array([[ 1.,  1.]], dtype=float32), array([[ 0.]], dtype=float32), 0.0]
    #
    # epochs = 3000
    # learningRate = 0.2



    epochs = 100000
    learningRate = 0.5
    momentum = 0.5

    with tf.name_scope('labels'):
        t = tf.placeholder(tf.float32, shape=[1, 1], name="labels")

    with tf.name_scope('inputLayer'):
        inputLayer = tf.placeholder(tf.float32, shape=[1, 2], name="inputLayer")

    with tf.name_scope('hiddenLayer_01'):
        W1 = tf.Variable(tf.random_uniform([2, 2], -1, 1), name="W1")
        # W1 = tf.Variable(tf.constant([
        #     [0.0553,    0.1319],
        #     [0.7538,    0.3559]
        # ]))
        bias1 = tf.Variable(tf.zeros([2]), name="Bias1")
        # bias1 = tf.Variable(tf.constant([0.3959, 0.8855]), name="Bias1")

        # x1 = tf.nn.relu(tf.matmul(inputLayer, W1) + bias1)
        x1 = tf.nn.sigmoid(tf.matmul(inputLayer, W1) + bias1)

        tf.summary.histogram('W1', W1)

    with tf.name_scope('hiddenLayer_02'):
        W2 = tf.Variable(tf.random_uniform([2, 2], -1, 1), name="W2")
        # W2 = tf.Variable(tf.constant([
        #     [0.0212,    0.2881],
        #     [0.8441,    0.2503]
        # ]))
        bias2 = tf.Variable(tf.zeros([2]), name="Bias2")
        # bias2 = tf.Variable(tf.constant([0.4884, 0.7290]), name="Bias2")

        # x2 = tf.nn.relu(tf.matmul(x1, W2) + bias2)
        x2 = tf.nn.sigmoid(tf.matmul(x1, W2) + bias2)

        tf.summary.histogram('W2', W2)

    with tf.name_scope('outputLayer'):
        W3 = tf.Variable(tf.random_uniform([2, 1], -1, 1), name="W3")
        # W3 = tf.Variable(tf.constant([
        #     [0.2026],
        #     [0.2163]
        # ]))
        bias3 = tf.Variable(tf.zeros([1]), name="Bias3")
        # bias3 = tf.Variable(tf.constant([0.9763]), name="Bias3")
        y = tf.matmul(x2, W3) + bias3
        tf.summary.histogram('W3', W3)

    with tf.name_scope('cost'):
        cost = tf.reduce_mean(tf.square(y - t) / tf.constant(2.0))
        # cost = tf.square(y - t) / tf.constant(2.0)
        tf.summary.scalar('cost', cost)

    train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)
    # train_step = tf.train.MomentumOptimizer(learningRate, momentum).minimize(cost)
    # train_step = tf.train.AdamOptimizer(learningRate).minimize(cost)




    start = -1
    step = 0.01
    end = 1 + step
    X01 = np.arange(start, end, step)
    X02 = np.arange(start, end, step)

    X01mesh, X02mesh = np.meshgrid(X01, X02)



    plotEvaluationInputList = list()
    for i in range(0, len(X01)):
        for j in range(0, len(X02)):
            plotEvaluationInputList.append([X01[i], X02[j]])


    plotLabelList = list()
    for inputPair in plotEvaluationInputList:
        if inputPair[0] == inputPair[1]:
            plotLabelList.append([0])
        else:
            plotLabelList.append([1])


    trainingData = list()
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == j:
                trainingData.append([[[i, j]], [[0]]])
            else:
                trainingData.append([[[i, j]], [[1]]])
    trainingData.append([[[-2, -2]], [[0]]])
    trainingData.append([[[2, 2]], [[0]]])

    print(trainingData)

    logDir = "C:\\tensorLogs\\xor"
    if not os.path.isdir(logDir):
        os.makedirs(logDir)
    dirs = os.listdir(logDir)
    runString = str(len(dirs) + 1).zfill(3)

    trainDir = os.path.join(logDir, "run{}".format(runString))



    saver = tf.train.Saver()






    mergedSummaries = tf.summary.merge_all()

    session = tf.Session()
    train_writer = tf.summary.FileWriter(trainDir, graph=session.graph)
    tf.global_variables_initializer().run(session=session)




    for i in range(1, epochs + 1):

        trainingDataIndex = (len(trainingData) + i) % len(trainingData)
        session.run(train_step, feed_dict={inputLayer: trainingData[trainingDataIndex][0], t: trainingData[trainingDataIndex][1]})

        if i == 1 or i % (epochs / 4) == 0:
            # pass
            print("\n++++++++++ epoch {} ++++++++++++++".format(i))
            print("y {}".format(session.run([inputLayer, y], feed_dict={inputLayer: trainingData[trainingDataIndex][0]})))
            print('W1 ', session.run(W1))
            print('B1 ', session.run(bias1))
            print('W2 ', session.run(W2))
            print('B2 ', session.run(bias2))
            print('W3 ', session.run(W3))
            print('B3 ', session.run(bias3))
            print('cost ', session.run(cost, feed_dict={inputLayer: trainingData[trainingDataIndex][0], t: trainingData[trainingDataIndex][1]}))

        if i == 1 or i % (epochs / 100) == 0 or i == epochs:
            summary = session.run(mergedSummaries, feed_dict={inputLayer: trainingData[trainingDataIndex][0], t: trainingData[trainingDataIndex][1]})
            train_writer.add_summary(summary, i)

    evaluations = list()
    for plotDataX0X1 in plotEvaluationInputList:
        evalData = [plotDataX0X1]
        evaluations.append(session.run([inputLayer, y], feed_dict={inputLayer: evalData}))

    evalData = [[[0, 0]], [[0]]]
    print(session.run([inputLayer, y, cost], feed_dict={inputLayer: evalData[0], t: evalData[1]}))
    evalData = [[[0, 1]], [[1]]]
    print(session.run([inputLayer, y, cost], feed_dict={inputLayer: evalData[0], t: evalData[1]}))
    evalData = [[[1, 0]], [[1]]]
    print(session.run([inputLayer, y, cost], feed_dict={inputLayer: evalData[0], t: evalData[1]}))
    evalData = [[[1, 1]], [[0]]]
    print(session.run([inputLayer, y, cost], feed_dict={inputLayer: evalData[0], t: evalData[1]}))



    # evaluate a profile
    xyProfile = np.arange(-1, 1.1, 0.02)
    xProfile = list()
    yProfile = list()
    for xyValue in xyProfile:
        xProfile.append(xyValue)
        yProfile.append(session.run([inputLayer, y], feed_dict={inputLayer: [[xyValue, -xyValue]]})[1][0])

    fig = plt.figure(1)
    ax = fig.add_subplot(211)

    thing = ax.plot(xProfile, yProfile, 'o')
    # surf = ax.plot_wireframe(X01, X02, ZPlot)

    # plt.show()

    ZPlot = np.zeros([len(X01), len(X02)])
    k = 0
    for i in range(0, len(X01)):
        for j in range(0, len(X02)):
            if k < len(X01)*len(X02):
                ZPlot[i][j] = evaluations[k][1]
                k += 1
            else:
                break




    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X01mesh, X02mesh, ZPlot, cmap='PiYG')
    # surf = ax.plot_wireframe(X01, X02, ZPlot)

    plt.show()



    savePath = os.path.join(trainDir, "savedSession")
    saver.save(session, savePath)

    train_writer.close()
    session.close()

    print("evaluating saved model:")
    newSession = tf.Session()
    saver.restore(newSession, savePath)

    evalData = [[[0.4, 0.4]], [[0]]]
    print(newSession.run([inputLayer, y, cost], feed_dict={inputLayer: evalData[0], t: evalData[1]}))
    newSession.close()

if __name__ == "__main__":
    run()