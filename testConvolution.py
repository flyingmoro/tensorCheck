# -*- encoding: utf-8 -*-

import os
import math
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


import tensorflow as tf
# tensorboard --logdir="C:\tensorLogs\xor"

from forwardNet import ForwardNet

from mySystem import MySystem



def run():
    h = 0.1
    model = MySystem(0, 0, h=h)
    tOfImpulseResponse, impulseResponse = model.getImpulseResponseOverTime()
    theNet = ForwardNet(impulseResponse=impulseResponse, h=h)

    session = tf.Session()
    tf.global_variables_initializer().run(session=session)

    duda = session.run(theNet.impulseMatrix)
    print(duda)

    timeOfTestSetPoints = list()
    testSetpoints = deque(maxlen=len(impulseResponse))
    for i in range(0, len(impulseResponse)):
        timeOfTestSetPoints.append(i * 0.01)
        testSetpoints.append([0.0])


    sysOutTime = list()
    sysOut = list()
    testRange = 200
    for n in range(0, testRange):
        testSetpoints.append([1.0])
        y = session.run(theNet.systemOutput, feed_dict={theNet.setPoints: testSetpoints})
        sysOutTime.append(n * 0.01)
        sysOut.append(y[0])

    import matplotlib.pyplot as plt

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    # ax1.set_xlim([0, 6])
    # ax1.set_ylim([-1, 1])
    ax1.plot(sysOutTime, sysOut)


    plt.show()




    session.close()


if __name__ == "__main__":
    run()