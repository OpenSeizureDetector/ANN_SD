#!/usr/bin/python

# Test the neural network usin greal data.

import os
import argparse
import numpy as np
import numpy.linalg as la
import scipy.signal
import matplotlib.pyplot as plt
from imutils import paths
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
import keras.utils
import keras.models
import keras.layers

PROC_FREQ = 25.  # Hz
SAMP_TIME = 5.0  # Seconds

HIDDEN_UNITS = 512
DROPOUT = 0.45
BATCH_SIZE = 128
EPOCHS = 200

def readFile(fname):
    """ Reads accelerometer data from file fname into a multi column numpy
    array - one row per data file row.
    It expects the file to be in the format produced by OpenSeizureDetector
    V3.0.4, which is a CSV file.
    Col 0 is the date
    Col 14 is the status text.
    Col 15 on is the raw data (in mg)
    """
    if (os.path.exists(fname)):
        # rawArr = np.genfromtxt(fname,delimiter=",", dtype=None)
        rawArr = np.loadtxt(fname,delimiter=",", dtype='string')
    else:
        print("readFile: ERROR - %s does not exist" % fname)
        rawArr = None

    #print(rawArr, rawArr.shape)
    #print(rawArr.shape)
    classArr = rawArr[:,0]
    print(classArr, classArr.shape)
    dataArr = rawArr[:,15:].astype(np.float)
    print(dataArr, dataArr.shape)
    #exit(-1)
    return classArr, dataArr




if (__name__ == "__main__"):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inDir", required=True,
	help="path to directory of input data files - one file per category"
        "with one row per 5 second training data sample")
    ap.add_argument("-m", "--model", required=True,
	            help="path to output model")
    ap.add_argument("-p", "--plot", dest='plot',action='store_true',
	            help="Plot the data as it is processed")
    args = vars(ap.parse_args())

    print args

    fileList = list(paths.list_files(args['inDir'],validExts=('.csv')))
    print(fileList)

    classArr = None
    dataArr = None
    
    for fname in fileList:
        fnameParts = fname.split("/")
        className = fnameParts[len(fnameParts)-1].split(".")[0]
        print("Reading file %s" % fname)
        print("className=%s" % className)
        cArr, dArr = readFile(fname)

        print(cArr, cArr.shape)
        print(dArr, dArr.shape)
        
        if (classArr is None):
            classArr = cArr
        else:
            print("classArr",classArr.shape, cArr.shape)
            classArr = np.hstack((classArr, cArr))
            print(classArr.shape)

        if (dataArr is None):
            dataArr = dArr
        else:
            print("dataArr",dataArr.shape, dArr.shape)
            dataArr = np.vstack((dataArr, dArr))
            print(dataArr.shape)

        print(classArr.shape, dataArr.shape)


    # Normalise the data
    dataArr = dataArr.astype('float32') - dataArr.mean()
    dataArr = dataArr / dataArr.max()
                
    if (args['plot']):
        # Plot 25 random datasets
        indices = np.random.randint(0, dataArr.shape[0], size=25)
        yData = dataArr[indices]
        labels = classArr[indices]
        xVals = np.linspace(0,
                           dataArr.shape[1]/PROC_FREQ,
                           dataArr.shape[1])
        plt.figure(figsize=(5,5))
        for i in range(len(indices)):
            plt.subplot(5,5, i+1)
            plt.plot(xVals,yData[i])
            plt.ylim((-1,1))
            plt.title(labels[i])
        plt.show()
    
    # initialize the model
    if (os.path.exists(args["model"])):
        print("Loading Model from disk")
        model = keras.models.load_model(args["model"])
    else:
        print("[ERROR] Model %s not found" % args["model"])
        exit(-1)

    model.summary()
    keras.utils.plot_model(model, to_file='test_model.png', show_shapes = True)

    #model.compile(loss='categorical_crossentropy',
    #              optimizer='adam',
    #              metrics=['accuracy'])


    predArr = model.predict_classes(dataArr)

    for i in range(len(dataArr)):
        print("%s predicted to be %s" % (classArr[i],predArr[i]))

