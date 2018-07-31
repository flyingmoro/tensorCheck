# -*- encoding: utf-8 -*-

import tensorflow as tf


class ForwardNet(object):
    def __init__(self, impulseResponse=list([0, 1, 2]), h=0.1):

        with tf.name_scope('labels'):
            self.t = tf.placeholder(tf.float32, shape=[1, 1], name="labels")

        # with tf.name_scope('inputLayer'):
        #     self.inputLayer = tf.placeholder(tf.float32, shape=[10, 1], name="inputLayer")
        #
        # with tf.name_scope('hiddenLayer_01'):
        #     self.W1 = tf.Variable(tf.random_uniform([10, 10], -1, 1), name="W1")
        #     # W1 = tf.Variable(tf.constant([
        #     #     [0.0553,    0.1319],
        #     #     [0.7538,    0.3559]
        #     # ]))
        #     self.bias1 = tf.Variable(tf.zeros([10, 1]), name="Bias1")
        #     # bias1 = tf.Variable(tf.constant([0.3959, 0.8855]), name="Bias1")
        #
        #     # x1 = tf.nn.relu(tf.matmul(inputLayer, W1) + bias1)
        #     self.x1 = tf.nn.relu(tf.matmul(self.W1, self.inputLayer) + self.bias1)


        with tf.name_scope('inputLayerU'):
            self.inputU = tf.placeholder(tf.float32, shape=[5, 1], name="inputU")

        with tf.name_scope('inputLayerY'):
            self.inputY = tf.placeholder(tf.float32, shape=[5, 1], name="inputY")


        with tf.name_scope('hiddenInputLayer_U1'):
            self.inputWU1 = tf.Variable(tf.random_uniform([5, 5], -1, 1), name="inputWU1")
            self.inputBiasU1 = tf.Variable(tf.zeros([5, 1]), name="inputBiasU1")
            self.inputU1 = tf.nn.relu(tf.matmul(self.inputWU1, self.inputU) + self.inputBiasU1)

        with tf.name_scope('hiddenInputLayer_Y1'):
            self.inputWY1 = tf.Variable(tf.random_uniform([5, 5], -1, 1), name="inputWY1")
            self.inputBiasY1 = tf.Variable(tf.zeros([5, 1]), name="inputBiasY1")
            self.inputY1 = tf.nn.relu(tf.matmul(self.inputWY1, self.inputY) + self.inputBiasY1)


        with tf.name_scope('hiddenInputLayer_U2'):
            self.inputWU2 = tf.Variable(tf.random_uniform([5, 5], -1, 1), name="inputWU2")
            self.inputBiasU2 = tf.Variable(tf.zeros([5, 1]), name="inputBiasU2")
            self.inputU2 = tf.nn.relu(tf.matmul(self.inputWU2, self.inputU1) + self.inputBiasU2)

        with tf.name_scope('hiddenInputLayer_Y2'):
            self.inputWY2 = tf.Variable(tf.random_uniform([5, 5], -1, 1), name="inputWY2")
            self.inputBiasY2 = tf.Variable(tf.zeros([5, 1]), name="inputBiasY2")
            self.inputY2 = tf.nn.relu(tf.matmul(self.inputWY2, self.inputY1) + self.inputBiasY2)

        # with tf.name_scope('combinedInputLayer'):
        #     self.input = tf.concat([self.inputU2, self.inputY2], 0)

        with tf.name_scope('combinedInputLayer'):
            self.input = tf.concat([self.inputU, self.inputY], 0)

        with tf.name_scope('hiddenLayer_01'):
            self.W1 = tf.Variable(tf.random_uniform([10, 10], -1, 1), name="W1")
            self.bias1 = tf.Variable(tf.zeros([10, 1]), name="Bias1")
            self.x1 = tf.nn.relu(tf.matmul(self.W1, self.input) + self.bias1)


        with tf.name_scope('hiddenLayer_02'):
            self.W2 = tf.Variable(tf.random_uniform([10, 10], -1, 1), name="W2")
            self.bias2 = tf.Variable(tf.zeros([10, 1]), name="Bias2")
            self.x2 = tf.nn.relu(tf.matmul(self.W2, self.x1) + self.bias2)


        with tf.name_scope('hiddenLayer_03'):
            self.combinedX1X2 = tf.concat([self.x1, self.x2], 0)
            self.W3 = tf.Variable(tf.random_uniform([20, 20], -1, 1), name="W3")
            self.bias3 = tf.Variable(tf.zeros([20, 1]), name="Bias3")
            self.x3 = tf.nn.relu(tf.matmul(self.W3, self.combinedX1X2) + self.bias3)


        with tf.name_scope('hiddenLayer_04'):
            self.W4 = tf.Variable(tf.random_uniform([20, 20], -1, 1), name="W4")
            self.bias4 = tf.Variable(tf.zeros([20, 1]), name="Bias4")
            self.x4 = tf.nn.relu(tf.matmul(self.W4, self.x3) + self.bias4)



        with tf.name_scope('outputLayer'):
            self.combinedDeeperLayers = tf.concat([self.x3, self.x4], 0)
            self.W5 = tf.Variable(tf.random_uniform([1, 40], -1, 1), name="W5")
            self.bias5 = tf.Variable(tf.zeros([1]), name="Bias5")
            self.y = tf.matmul(self.W5, self.combinedDeeperLayers) + self.bias5


        with tf.name_scope("system"):
            impulseResponseLength = len(impulseResponse)
            impulseMatrix = list()

            # # matrix for calculating future points of ode
            # for rowCounter in range(0, impulseResponseLength):
            #     impulseMatrix.append(list())
            #     for columnCounter in range(0, impulseResponseLength):
            #         if rowCounter <= columnCounter:
            #             impulseResponseIndex = rowCounter + impulseResponseLength - columnCounter - 1
            #             impulseMatrix[rowCounter].append(float(impulseResponse[impulseResponseIndex]))
            #         else:
            #             impulseMatrix[rowCounter].append(0.0)

            # vector for calculation of next point of ode
            impulseMatrix.append(list())
            for columnCounter in range(0, impulseResponseLength):
                impulseMatrix[0].append(impulseResponse[impulseResponseLength - 1 - columnCounter])

            self.impulseMatrix = tf.constant(impulseMatrix, name="impulseMatrix", dtype=tf.float32)
            self.setPoints = tf.Variable(tf.zeros([impulseResponseLength, 1]), name="setPoints")
            self.systemOutput = tf.matmul(self.impulseMatrix, self.setPoints) * tf.constant(h)
            self.firstImpulseResponseValue = tf.constant(impulseResponse[0], name="firstImpulseResponseValue")


        with tf.name_scope('cost'):
            self.cost = tf.square(self.t - self.y) / tf.constant(2.0)
            self.trainingCost = tf.reduce_mean(tf.square(self.t - self.y) / tf.constant(2.0))
            self.meanCost = tf.reduce_mean(self.cost)

            self.costToSystemResponse = tf.square(self.t - (self.systemOutput + self.y)) / tf.constant(2.0)

            # self.costSpecial = tf.square(self.y - self.t) / tf.constant(2.0)

