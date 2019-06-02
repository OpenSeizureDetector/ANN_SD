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
import keras.callbacks

import sklearn
print(sklearn.__version__)
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn.manifold import TSNE

import sys
import platform

#import accUtil
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
LAYER_FILTERS = [32, 64] # No. of CNN layers and filters per layer
KERNEL_SIZE = 3
LATENT_DIM = 250
ACTIVATION = 'tanh'

def logV(msg):
    if (DEBUG>=3):
        print(msg)



def buildModel(inSize):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(inSize,
                                  input_dim=inSize,
                                  activation=ACTIVATION))
    model.add(keras.layers.Dense(LATENT_DIM*4,
                                  activation=ACTIVATION))
    model.add(keras.layers.Dense(LATENT_DIM*2,
                                  activation=ACTIVATION))
    model.add(keras.layers.Dense(LATENT_DIM,
                                 # input_dim=inSize,
                                 activation=ACTIVATION,
                                 activity_regularizer=keras.regularizers.l1(10e-5),
                                 name='latent'))
    model.add(keras.layers.Dense(LATENT_DIM*2,
                                 activation=ACTIVATION))
    model.add(keras.layers.Dense(LATENT_DIM*4,
                                 activation=ACTIVATION))
    model.add(keras.layers.Dense(inSize, activation=ACTIVATION))
    model.summary()
    return model

def buildModel_2(inSize):
    model = keras.models.Sequential()
    #model.add(keras.layers.Dense(inSize,
    #                              input_dim=inSize,
    #                              activation=ACTIVATION))
    model.add(keras.layers.Dense(LATENT_DIM,
                                 input_dim=inSize,
                                 activation=ACTIVATION,
                                 activity_regularizer=keras.regularizers.l1(10e-5),
                                 name='latent'))
    model.add(keras.layers.Dense(inSize, activation=ACTIVATION))
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
    ap.add_argument("-e", "--epochs", required=False, default=50,
                    help="Number of epochs to train")
    ap.add_argument("-p", "--plot", dest='plot', action='store_true',
                    help="Plot the data as it is processed")
    ap.add_argument("-t", "--tsne", dest='tsne', action='store_true',
                    help="Plot TSNE representation")
    args = vars(ap.parse_args())

    print(args)

    nSamp = args['nSamp']
    nEpochs = int(args['epochs'])

    if (os.path.exists(args["model"])):
        print("Loading Model from disk")
        model = keras.models.load_model(args["model"])
        model.summary()
    else:
        model = buildModel(nSamp)
    
    fname = args['inFile']

    timeArr, hrArr, accArr, \
        fftArr, fftFreq = osdDataUtils.getOsdData("TestData/Normal",
                                                  nSamp)

    print("accArr", accArr.shape, accArr.dtype, accArr[0, 0].dtype)
    print("fftArr", fftArr.shape)
    print("fftFreq", fftFreq, fftFreq.shape)


    # normalise accArr to be constrained to -1 to +1
    normVal = max(accArr.max(), abs(accArr.min()))
    accArr = accArr / normVal

    # Now split our data for training, test and validation
    # We use 10% of the data for post-fit testing
    # then 30% of the remainder for validation during training.

    accArr_t, accArr_test = train_test_split(accArr,test_size = 0.1)
    accArr_train, accArr_val = train_test_split(accArr_t,test_size = 0.3)

    print("accArr_train",accArr_train.shape)
    print("accArr_val",accArr_val.shape)
    print("accArr_test",accArr_test.shape)

    logV("Compiling Model")
    model.compile(optimizer='adam', loss='mse')

    # Set up Keras to save the best model in case it deteriorates
    #    with more training
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                  patience=10,
                                  verbose=1,
                                  mode='min')
    mcp_save = keras.callbacks.ModelCheckpoint('autoencoder_anomaly_best.hdf5',
                                               save_best_only=True,
                                               monitor='val_loss',
                                               verbose=1,
                                               mode='min')
    reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=7,
        verbose=1,
        epsilon=1e-4,
        mode='min')


    logV("Fitting Model")
    history = model.fit(accArr_train,accArr_train,
                        validation_data = (accArr_val, accArr_val),
                        epochs = nEpochs,
                        batch_size = BATCH_SIZE,
                        shuffle = True,
                        callbacks=[earlyStopping, mcp_save, reduce_lr_loss])
    # save the model to disk
    logV("[INFO] serializing network...")
    model.save(args["model"])

    print(history)
    
    test_decoded = model.predict(accArr_test)


    if (args['plot']):
        print("Plotting Sample of test data")
        # Plot 25 random datasets
        indices = np.random.randint(0, len(test_decoded[0]), size=nSamp)
        yData = accArr_test[indices]
        yFit = model.predict(yData)
        print("yData.shape=",yData.shape)
        print("yFit.shape=",yFit.shape)
        xVals = np.linspace(0,
                           len(accArr_test[0]),
                            len(accArr_test[0]))
        plt.figure(figsize=(5,5))
        for i in range(len(indices)):
            plt.subplot(5,5, i+1)
            plt.plot(xVals,yData[i],yFit[i])
            #plt.ylim((-1,1))
        plt.show()


    if (args['tsne']):
        tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
        tsneData = tsne_model.fit_transform(encOut)
        print(tsneData)
        fig, ax = plt.subplots()
        ax.scatter(tsneData[:,0], tsneData[:,1])
        plt.show()
        


    
