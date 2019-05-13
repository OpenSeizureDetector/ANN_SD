#!/usr/bin/env python

# Test a pre-trained network on real data from OSD Watch App

import os
import argparse
import numpy as np
# import numpy.linalg as la
# import scipy.signal
import matplotlib.pyplot as plt
import keras.utils
import keras.models
import keras.layers

import sklearn
from sklearn.model_selection import train_test_split
import sklearn.preprocessing

import sys
import platform
print("Running Python%s - V %s" %
      (sys.version_info[0], platform.python_version()))
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")
print(sklearn.__version__)


PROC_FREQ = 25.  # Hz
SAMP_TIME = 5.0  # Seconds
NSAMP = 20
NORM_MAX = 2000.  # mG FSD


def getData(fHandle):
    """ Reads one line of accelerometer data from the open file
    fHandle.   Returns an array of accelerometer data and a string
    of the time of the data.
    @returns timeStr, accArr
    """
    accStartCol = 15
    accNumSamp = 125
    # hrCol = 14
    
    accArr = []
    timeStr = ""
    lineStr = fHandle.readline()
    if (lineStr != ""):
        lParts = lineStr.split(",")
        timeStr = lParts[0]
        for i in range(accStartCol, accStartCol+accNumSamp):
            accArr.append(float(lParts[i]))
        # print("accArr: ",accArr)
    return timeStr, accArr


if (__name__ == "__main__"):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inFile", required=True,
                    help="input data file")
    ap.add_argument("-m", "--model", required=True,
                    help="path to output model")
    ap.add_argument("-p", "--plot", dest='plot', action='store_true',
                    help="Plot the data as it is processed")
    args = vars(ap.parse_args())

    print(args)

    classArr = None
    dataArr = None

    if (os.path.exists(args["model"])):
        print("Loading Model from disk")
        model = keras.models.load_model(args["model"])
        model.summary()
    else:
        print("***ERRROR - Model %s does not exist" % model)
        exit(-1)
    
    fname = args['inFile']
    tStr = "start"
    dataArr = []
    fig = None
    ax = None
    line = None
    if (os.path.exists(fname)):
        fHandle = open(fname, "r")

        while (tStr != ""):
            tStr, dArr = getData(fHandle)
            # print(tStr)
            dataArr.extend(dArr)
            sampArr = []
            while(len(dataArr) >= NSAMP):
                # print("len(samp)=%d, len(dataArr)=%d"
                #      % (len(sampArr),len(dataArr)))
                sampArr = dataArr[:NSAMP]
                dataArr = dataArr[NSAMP:]
                # print("len(samp)=%d, len(dataArr)=%d"
                #      % (len(sampArr),len(dataArr)))

                # Normalise the data
                sampArr = np.asarray([sampArr])
                sampArr = sampArr.astype('float32') - sampArr.mean()
                sampArr = sampArr / NORM_MAX
                # print("sampArr=",sampArr, sampArr.shape)
                prob = model.predict(sampArr)
                print(tStr, prob)

                if (args['plot']):
                    yData = sampArr[0]
                    xVals = np.linspace(0,
                                        sampArr.shape[1]/PROC_FREQ,
                                        sampArr.shape[1])
                    if (fig == None):
                        plt.ion()
                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        line1, = ax.plot(xVals, yData, 'b-')
                        ax.set_ylim(-1, 1)
                    line1.set_ydata(yData)
                    fig.canvas.draw()
                    # plt.figure()
                    # plt.plot(xVals, yData)
                    # plt.ylim((-1, 1))
                    # plt.title(tStr)
                    # plt.show()

        print("End of File")
        exit(-1)

    else:
        print("readFile: ERROR - %s does not exist" % fname)
        exit(-1)
    
    
    # initialize the model
    if (os.path.exists(args["model"])):
        print("Loading Model from disk")
        model = keras.models.load_model(args["model"])
    else:
        print("[INFO] creating model...data_train.shape=",data_train.shape)
        print("Data Train: ",data_train)
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(HIDDEN_UNITS, input_dim=data_train.shape[1]))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(DROPOUT))
        model.add(keras.layers.Dense(HIDDEN_UNITS))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(DROPOUT))
        model.add(keras.layers.Dense(len(unique)))
        model.add(keras.layers.Activation('softmax'))

    model.summary()
    keras.utils.plot_model(model, to_file='model.png', show_shapes = True)

    filepath="acc_ann-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    H = model.fit(data_train, label_train_cat,
              validation_data=(data_test, label_test_cat),
              epochs = EPOCHS, batch_size=BATCH_SIZE,
              callbacks = callbacks_list
    )

    # save the model to disk
    print("[INFO] serializing network...")
    model.save(args["model"])
    
    loss, acc = model.evaluate(data_test, label_test_cat, batch_size = BATCH_SIZE)
    print("\nTest accuracy: %0.1f%%" % (100. * acc))

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy for Farm identification")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("training.png")

