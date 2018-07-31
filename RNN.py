# -*- encoding: utf-8 -*-
import os

import tensorflow as tf
# tensorboard --logdir="C:\tensorLogs"


def run():

    with tf.name_scope('chekko'):
        node1 = tf.placeholder(tf.float32, name="node1")
        node2 = tf.placeholder(tf.float32, name="node2")
        node3 = tf.add(node1, node2, name="node3")

    tf.summary.scalar('node1', node1)

    with tf.name_scope('superGraph'):
        triple = tf.multiply(node3, 3, name="triple")

    tf.summary.scalar('result', triple)

    merged = tf.summary.merge_all()


    with tf.Session() as session:
        # summary = session.run(node3, {node1: [3, 4.5], node2: 4.5})

        ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__))
        logDir = "C:\\tensorLogs\\example"
        graphDir = os.path.join(logDir, "graphs")
        scalarDir = os.path.join(logDir, "scalar")
        writer = tf.summary.FileWriter(graphDir, graph=session.graph)
        writerOne = tf.summary.FileWriter(scalarDir)

        tf.global_variables_initializer().run()

        for i in range(0, 10):
            summary = session.run(merged, {node1: i, node2: i*2})
            writerOne.add_summary(summary, i)
            print(summary)

        writer.close()


if __name__ == "__main__":
    run()