#!/usr/bin/env python

""" First go at using an autoencoder to classify 
    unlabeled accelerometer data.
    We take 5 second samples of accelerometer data from openseizuredetector
    and run it through an auto encoder that produces a 2d vector
    to represent the acceleration state.
    Then we plot all the 2d vectors to see if they are clustering or not...
"""


import os
import argparse
import numpy as np
import numpy.linalg as la
import scipy.signal
import matplotlib.pyplot as plt
import keras.utils
import keras.models
import keras.layers
import keras.backend
import keras.regularizers

import sklearn
print(sklearn.__version__)
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn.manifold import TSNE
import pandas as pd

import sys
import platform

import accUtil

print("Running Python%s - V %s" % (sys.version_info[0],platform.python_version()))
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")


##################################################################
# Fixed Parameters
##################################################################
DEBUG = 3

PROC_FREQ = 25.  # Hz
SAMP_TIME = 5.0  # Seconds
NSAMP = 120  # Number of samples to use for input to autoencoder
NORM_MAX = 2000.  # mG FSD

HIDDEN_UNITS = 32
DROPOUT = 0.45
BATCH_SIZE = 128
EPOCHS = 50
LAYER_FILTERS = [32, 64] # No. of CNN layers and filters per layer
KERNEL_SIZE = 3
LATENT_DIM = 32

def logV(msg):
    if (DEBUG>=3):
        print(msg)


def buildModel_f(inSize):
    """ Return a Keras Model to use for the autoencoder.
    It expects the model input to be a vector of size inSize.
    """
    logV("Build Model - inSize = %d" % inSize)
    # first build the encoder model
    inputs = keras.layers.Input(shape=(inSize,), name='encoder_input')
    x = inputs
    # generate the latent vector
    latent = keras.layers.Dense(LATENT_DIM,
                                activation='relu',
                                activity_regularizer=keras.regularizers.l1(10e-5),
                                name='latent_vector')(x)
    encoder = keras.models.Model(inputs, latent, name='encoder')
    encoder.summary()

    # build the decoder model
    latent_inputs = keras.layers.Input(shape=(LATENT_DIM,),
                                       name='decoder_input')
    x = keras.layers.Dense(inSize)(latent_inputs)
    outputs = x

    # instantiate decoder model
    decoder = keras.models.Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    # autoencoder = encoder + decoder
    # instantiate autoencoder model
    autoencoder = keras.models.Model(
        inputs,
        decoder(encoder(inputs)),
        name='autoencoder')
    autoencoder.summary()

    return autoencoder


def buildModel(inSize):
    model = keras.models.Sequential()
    # model.add(keras.layers.Dense(inSize,
    #                              input_dim=inSize,
    #                              activation='relu'))
    # model.add(keras.layers.Dense(LATENT_DIM*4,
    #                              activation='relu'))
    # model.add(keras.layers.Dense(LATENT_DIM*2,
    #                              activation='relu'))
    model.add(keras.layers.Dense(LATENT_DIM,
                                 input_dim=inSize,
                                 activation='relu',
                                 activity_regularizer=keras.regularizers.l1(10e-5),
                                 name='latent'))
    # model.add(keras.layers.Dense(LATENT_DIM*2,
    #                              activation='relu'))
    # model.add(keras.layers.Dense(LATENT_DIM*4,
    #                              activation='relu'))
    model.add(keras.layers.Dense(inSize, activation='relu'))
    model.summary()
    return model

def getEncoderOutput(model, testData):
    encoder = keras.models.Model(inputs=model.input,
                                 outputs=model.get_layer('latent').output)
    encOutput = encoder.predict(testData)
    return encOutput

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

########################################################################
# Main Program
########################################################################
if (__name__ == "__main__"):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inFile", required=True,
                    help="input data file")
    ap.add_argument("-m", "--model", required=True,
                    help="path to output model")
    ap.add_argument("-p", "--plot", dest='plot', action='store_true',
                    help="Plot the data as it is processed")
    ap.add_argument("-t", "--tsne", dest='tsne', action='store_true',
                    help="Plot TSNE representation")
    args = vars(ap.parse_args())

    print(args)

    classArr = None
    dataArr = None

    if (os.path.exists(args["model"])):
        print("Loading Model from disk")
        model = keras.models.load_model(args["model"])
        model.summary()
    else:
        model = buildModel(NSAMP)
    
    fname = args['inFile']
    tStr = "start"
    dataArr = []
    allDataArr = None
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

                if (allDataArr is None):
                    allDataArr = sampArr
                else:
                    # print("allDataArr",allDataArr.shape, sampArr.shape)
                    allDataArr = np.vstack((allDataArr, sampArr))
                    # print("allDataArr.shape",allDataArr.shape)


                # prob = model.predict(sampArr)
                # print(tStr, prob)

                # if (args['plot']):
                #     yData = sampArr[0]
                #     xVals = np.linspace(0,
                #                         sampArr.shape[1]/PROC_FREQ,
                #                         sampArr.shape[1])
                #     if (fig == None):
                #         plt.ion()
                #         fig = plt.figure()
                #         ax = fig.add_subplot(111)
                #         line1, = ax.plot(xVals, yData, 'b-')
                #         ax.set_ylim(-1, 1)
                #     line1.set_ydata(yData)
                #     fig.canvas.draw()
                #     # plt.figure()
                #     # plt.plot(xVals, yData)
                #     # plt.ylim((-1, 1))
                #     # plt.title(tStr)
                #     # plt.show()

        print("End of File - allDataArr = ", allDataArr.shape)
        # allDataArr = allDataArr.reshape((
        #     allDataArr.shape[0],
        #     allDataArr.shape[1],
        #     1))
        print("Reshaped array - allDataArr = ", allDataArr.shape)

        logV("Compiling Model")
        model.compile(optimizer='adam', loss='mse')
        logV("Fitting Model")
        model.fit(allDataArr,allDataArr,
                  epochs = EPOCHS,
                  batch_size = BATCH_SIZE,
                  shuffle = True)
        # save the model to disk
        logV("[INFO] serializing network...")
        model.save(args["model"])


        if (args['plot']):
            yData = allDataArr[0:1,:]
            logV("len(yData)=%d" % len(yData))
            print("yData", yData.shape, yData)
            print("yData[0]", yData[0].shape, yData[0])
            yFit = model.predict(yData)
            print("yFit",yFit.shape, yFit)
            xVals = np.linspace(0,len(yData[0]),len(yData[0]))
            fig, ax = plt.subplots()
            print("xVals",xVals.shape, xVals)
            ax.plot(xVals,yData[0],yFit[0])
            plt.show()

            encOut = getEncoderOutput(model,allDataArr)
            fig, ax = plt.subplots()
            ax.scatter(encOut[:,0], encOut[:,1])
            plt.show()
        
            print("Plotting Sample of data")
            # Plot 25 random datasets
            indices = np.random.randint(0, len(allDataArr[0]), size=25)
            yData = allDataArr[indices]
            yFit = model.predict(yData)
            print("yData.shape=",yData.shape)
            print("yFit.shape=",yFit.shape)
            xVals = np.linspace(0,
                               len(allDataArr[0]),
                                len(allDataArr[0]))
            plt.figure(figsize=(5,5))
            for i in range(len(indices)):
                plt.subplot(5,5, i+1)
                plt.plot(xVals,yData[i],yFit[i])
                plt.ylim((-1,1))
            plt.show()


        if (args['tsne']):
            tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
            tsneData = tsne_model.fit_transform(encOut)
            print(tsneData)
            fig, ax = plt.subplots()
            ax.scatter(tsneData[:,0], tsneData[:,1])
            plt.show()
        

    else:
        print("readFile: ERROR - %s does not exist" % fname)
        exit(-1)
    
