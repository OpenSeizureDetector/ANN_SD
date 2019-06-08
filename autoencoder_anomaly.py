#!/usr/bin/env python

""" 
Autoencoder based anomaly detector.   Trains an autoencoder neural network
on 'normal' data, then checks the reconstruction error for test data.
If the reconstruction error is greater than a threshold, it treats it
as anomalous.
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
import osdDataUtils

print("Running Python%s - V %s" % (sys.version_info[0],platform.python_version()))
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")


##################################################################
# Fixed Parameters
##################################################################
DEBUG = 3


HIDDEN_UNITS = 32
DROPOUT = 0.45
BATCH_SIZE = 128
EPOCHS = 50
LAYER_FILTERS = [32, 64] # No. of CNN layers and filters per layer
KERNEL_SIZE = 3
LATENT_DIM = 250

def logV(msg):
    if (DEBUG>=3):
        print(msg)



def buildModel(inSize):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(inSize,
                                  input_dim=inSize,
                                  activation='relu'))
    # model.add(keras.layers.Dense(LATENT_DIM*4,
    #                               activation='relu'))
    model.add(keras.layers.Dense(LATENT_DIM/2,
                                  activation='relu'))
    model.add(keras.layers.Dense(LATENT_DIM,
                                 input_dim=inSize,
                                 activation='relu',
                                 activity_regularizer=keras.regularizers.l1(10e-5),
                                 name='latent'))
    model.add(keras.layers.Dense(LATENT_DIM/2,
                                 activation='relu'))
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
    ap.add_argument("-n", "--nSamp", required=False, default=25,
                    help="Number of samples per analysis period")
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
        model = buildModel(args['nSamp'])
    
    fname = args['inFile']

    timeArr, hrArr, accArr, \
        fftArr, fftFreq = osdDataUtils.getOsdData("TestData/Normal",
                                                  args['nSamp'])

    print("accArr", accArr.shape, accArr.dtype, accArr[0, 0].dtype)
    print("fftArr", fftArr.shape)
    print("fftFreq", fftFreq, fftFreq.shape)


    logV("Compiling Model")
    model.compile(optimizer='adam', loss='mse')
    logV("Fitting Model")
    model.fit(accArr,accArr,
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
        


    
