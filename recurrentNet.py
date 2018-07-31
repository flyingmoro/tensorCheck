# -*- encoding: utf-8 -*-

import tensorflow as tf


class RecurrentNet(object):
    def __init__(self, impulseResponse=list([0, 1, 2]), h=0.1):

        with tf.name_scope('labels'):
            self.t = tf.placeholder(tf.float32, shape=[1, 1], name="labels")

        with tf.name_scope('inputLayer'):
            self.input = tf.placeholder(tf.float32, shape=[2, 1], name="input")


        with tf.name_scope('hiddenLayer_01'):
            self.W1 = tf.Variable(tf.random_uniform([2, 2], -1, 1), name="W1")
            self.WRec1 = tf.Variable(tf.random_uniform([2, 1], -1, 1), name="WRec1")
            self.history_01 = tf.placeholder(tf.float32, shape=[2, 1], name="history_01")
            self.bias1 = tf.Variable(tf.random_uniform([2, 1], -1, 1), name="Bias1")
            self.x1 = tf.nn.relu(tf.matmul(self.W1, self.input) + self.bias1 + tf.multiply(self.WRec1, self.history_01))

        with tf.name_scope('hiddenLayer_02'):
            self.W2 = tf.Variable(tf.random_uniform([2, 2], -1, 1), name="W2")
            self.WRec2 = tf.Variable(tf.random_uniform([2, 1], -1, 1), name="WRec2")
            self.history_02 = tf.placeholder(tf.float32, shape=[2, 1], name="history_02")
            self.bias2 = tf.Variable(tf.random_uniform([2, 1], -1, 1), name="Bias2")
            self.x2 = tf.nn.relu(tf.matmul(self.W2, self.x1) + self.bias2 + tf.multiply(self.WRec2, self.history_02))

        with tf.name_scope('hiddenLayer_03'):
            self.W3 = tf.Variable(tf.random_uniform([2, 2], -1, 1), name="W3")
            self.WRec3 = tf.Variable(tf.random_uniform([2, 1], -1, 1), name="WRec3")
            self.history_03 = tf.placeholder(tf.float32, shape=[2, 1], name="history_03")
            self.bias3 = tf.Variable(tf.random_uniform([2, 1], -1, 1), name="Bias3")
            self.x3 = tf.nn.relu(tf.matmul(self.W3, self.x2) + self.bias3 + tf.multiply(self.WRec3, self.history_03))

        with tf.name_scope('hiddenLayer_04'):
            self.W4 = tf.Variable(tf.random_uniform([2, 2], -1, 1), name="W4")
            self.WRec4 = tf.Variable(tf.random_uniform([2, 1], -1, 1), name="WRec4")
            self.history_04 = tf.placeholder(tf.float32, shape=[2, 1], name="history_04")
            self.bias4 = tf.Variable(tf.random_uniform([2, 1], -1, 1), name="Bias4")
            self.x4 = tf.nn.relu(tf.matmul(self.W4, self.x3) + self.bias4 + tf.multiply(self.WRec4, self.history_04))

        with tf.name_scope('hiddenLayer_05'):
            self.W5 = tf.Variable(tf.random_uniform([2, 2], -1, 1), name="W5")
            self.WRec5 = tf.Variable(tf.random_uniform([2, 1], -1, 1), name="WRec5")
            self.history_05 = tf.placeholder(tf.float32, shape=[2, 1], name="history_05")
            self.bias5 = tf.Variable(tf.random_uniform([2, 1], -1, 1), name="Bias5")
            self.x5 = tf.nn.relu(tf.matmul(self.W5, self.x4) + self.bias5 + tf.multiply(self.WRec5, self.history_05))

        with tf.name_scope('hiddenLayer_06'):
            self.W6 = tf.Variable(tf.random_uniform([2, 2], -1, 1), name="W6")
            self.WRec6 = tf.Variable(tf.random_uniform([2, 1], -1, 1), name="WRec6")
            self.history_06 = tf.placeholder(tf.float32, shape=[2, 1], name="history_06")
            self.bias6 = tf.Variable(tf.random_uniform([2, 1], -1, 1), name="Bias6")
            self.x6 = tf.matmul(self.W6, self.x5) + self.bias6 + tf.multiply(self.WRec6, self.history_06)


        with tf.name_scope('outputLayer'):
            self.WOut = tf.Variable(tf.random_uniform([1, 2], -1, 1), name="W_out")
            self.biasOut = tf.Variable(tf.random_uniform([1], -1, 1), name="Bias_out")
            self.y = tf.matmul(self.WOut, self.x6) + self.biasOut

        with tf.name_scope('cost'):
            self.cost = tf.square(self.t - self.y) / tf.constant(2.0)
            self.trainingCost = tf.reduce_mean(self.cost)


