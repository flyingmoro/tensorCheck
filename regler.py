# -*- encoding: utf-8 -*-
import os

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
# tensorboard --logdir="C:\tensorLogs\xor"



def run():
    epochs = 100000
    learningRate = 0.5

    with tf.name_scope('labels'):
        t = tf.placeholder(tf.float32, shape=[1, 1], name="labels")

    with tf.name_scope('inputLayer'):
        inputLayer = tf.placeholder(tf.float32, shape=[10, 1], name="inputLayer")

    with tf.name_scope('hiddenLayer_01'):
        W1 = tf.Variable(tf.random_uniform([4, 10], -1, 1), name="W1")
        # W1 = tf.Variable(tf.constant([
        #     [0.0553,    0.1319],
        #     [0.7538,    0.3559]
        # ]))
        bias1 = tf.Variable(tf.zeros([4]), name="Bias1")
        # bias1 = tf.Variable(tf.constant([0.3959, 0.8855]), name="Bias1")

        # x1 = tf.nn.relu(tf.matmul(inputLayer, W1) + bias1)
        x1 = tf.nn.sigmoid(tf.matmul(inputLayer, W1) + bias1)

        tf.summary.histogram('W1', W1)

    with tf.name_scope('hiddenLayer_02'):
        W2 = tf.Variable(tf.random_uniform([2, 4], -1, 1), name="W2")
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
        W3 = tf.Variable(tf.random_uniform([1, 2], -1, 1), name="W3")
        # W3 = tf.Variable(tf.constant([
        #     [0.2026],
        #     [0.2163]
        # ]))
        bias3 = tf.Variable(tf.zeros([1]), name="Bias3")
        # bias3 = tf.Variable(tf.constant([0.9763]), name="Bias3")
        y = tf.nn.sigmoid(tf.matmul(x2, W3) + bias3)
        tf.summary.histogram('W3', W3)

    with tf.name_scope('cost'):
        cost = tf.reduce_mean(tf.square(y - t) / tf.constant(2.0))
        # cost = tf.square(y - t) / tf.constant(2.0)
        tf.summary.scalar('cost', cost)

    train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)
    # train_step = tf.train.MomentumOptimizer(learningRate, momentum).minimize(cost)
    # train_step = tf.train.AdamOptimizer(learningRate).minimize(cost)




    logDir = "C:\\tensorLogs\\regler"
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


    savePath = os.path.join(trainDir, "savedSession")
    saver.save(session, savePath)

    train_writer.close()
    session.close()

    # print("evaluating saved model:")
    # newSession = tf.Session()
    # saver.restore(newSession, savePath)
    #
    # evalData = [[[0.4, 0.4]], [[0]]]
    # print(newSession.run([inputLayer, y, cost], feed_dict={inputLayer: evalData[0], t: evalData[1]}))
    # newSession.close()

if __name__ == "__main__":
    run()