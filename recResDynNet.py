# -*- encoding: utf-8 -*-

import tensorflow as tf


class RecurrentResDynNet(object):
    def __init__(self, batchSize=1, hiddenLayerCount=5, learningRate=0.1, impulseResponse=list([0, 1, 2]), h=0.1):
        self.batchSize = batchSize
        self.hiddenLayerCount = hiddenLayerCount

        with tf.name_scope('labels'):
            self.t = tf.placeholder(tf.float32, shape=[1, self.batchSize], name="labels")


        self.allLayerOps = list()

        with tf.name_scope('inputLayer'):
            thisLayer = dict()
            thisLayer["output"] = tf.placeholder(tf.float32, shape=[2, self.batchSize], name="input")
            self.allLayerOps.append(thisLayer)


        with tf.name_scope('hiddenLayer_01'):
            thisLayer = dict()
            previousLayer = self.allLayerOps[-1]
            thisLayer["W"] = tf.Variable(tf.random_uniform([2, 2], -1, 1), name="W1")
            thisLayer["WRecurrent"] = tf.Variable(tf.random_uniform([2, 1], -1, 1), name="WRec1")
            thisLayer["history"] = tf.placeholder(tf.float32, shape=[2, self.batchSize], name="history_01")
            thisLayer["bias"] = tf.Variable(tf.random_uniform([2, 1], -1, 1), name="Bias1")
            thisLayer["output"] = tf.nn.relu(tf.matmul(thisLayer["W"], previousLayer["output"]) + thisLayer["bias"] + tf.multiply(thisLayer["WRecurrent"], thisLayer["history"]))
            self.allLayerOps.append(thisLayer)

        for hiddenLayerNumber in range(2, hiddenLayerCount + 1):
            with tf.name_scope("hiddenLayer_{}".format(hiddenLayerNumber)):
                thisLayer = dict()
                previousPreviousLayer = self.allLayerOps[-2]
                previousLayer = self.allLayerOps[-1]
                thisLayer["input"] = tf.concat([previousPreviousLayer["output"], previousLayer["output"]], 0)
                layerWidth = int(thisLayer["input"].shape[0])
                print(layerWidth)
                thisLayer["W"] = tf.Variable(tf.random_uniform([layerWidth, layerWidth], -1, 1), name="W_{}".format(hiddenLayerNumber))
                thisLayer["WRecurrent"] = tf.Variable(tf.random_uniform([layerWidth, 1], -1, 1), name="WRec_{}".format(hiddenLayerNumber))
                thisLayer["history"] = tf.placeholder(tf.float32, shape=[layerWidth, self.batchSize], name="history_{}".format(hiddenLayerNumber))
                thisLayer["bias"] = tf.Variable(tf.random_uniform([layerWidth, 1], -1, 1), name="Bias_{}".format(hiddenLayerNumber))
                thisLayer["output"] = tf.nn.relu(tf.matmul(thisLayer["W"], thisLayer["input"]) + thisLayer["bias"] + tf.multiply(thisLayer["WRecurrent"], thisLayer["history"]))
                self.allLayerOps.append(thisLayer)



        with tf.name_scope('outputLayer'):
            thisLayer = dict()
            previousLayer = self.allLayerOps[-1]
            layerWidth = int(previousLayer["output"].shape[0])
            thisLayer["W"] = tf.Variable(tf.random_uniform([1, layerWidth], -1, 1), name="W_out")
            thisLayer["bias"] = tf.Variable(tf.random_uniform([1], -1, 1), name="Bias_out")
            # thisLayer["output"] = tf.matmul(self.WOut, self.x6) + self.biasOut
            thisLayer["output"] = tf.sigmoid(tf.matmul(thisLayer["W"], previousLayer["output"]) + thisLayer["bias"])
            self.allLayerOps.append(thisLayer)


        with tf.name_scope('training'):
            lastLayer = self.allLayerOps[-1]
            self.cost = tf.square(self.t - lastLayer["output"]) / tf.constant(2.0)
            self.trainingCost = tf.reduce_mean(self.cost)
            tf.summary.scalar('trainingCost', self.trainingCost)
            self.optimizer = tf.train.AdagradOptimizer(learningRate)
            self.train_step = self.optimizer.minimize(self.trainingCost)
            self.gradients = self.optimizer.compute_gradients(self.trainingCost)

        with tf.name_scope('weights'):
            for i, layer in enumerate(self.allLayerOps):
                if not "W" in layer:
                    continue
                tf.summary.histogram("W_{}".format(i), layer["W"])

        with tf.name_scope('bias'):
            for i, layer in enumerate(self.allLayerOps):
                if not "bias" in layer:
                    continue
                tf.summary.histogram("bias_{}".format(i), layer["bias"])

        with tf.name_scope('weights_recurrent'):
            for i, layer in enumerate(self.allLayerOps):
                if not "WRecurrent" in layer:
                    continue
                tf.summary.histogram("weights_rec__{}".format(i), layer["WRecurrent"])
