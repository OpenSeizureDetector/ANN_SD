#!/usr/bin/python

# Train a dense neural network with accelerometer data.

import os
import argparse
import numpy as np
import numpy.linalg as la
import scipy.signal
import matplotlib.pyplot as plt
from imutils import paths
from sklearn.model_selection import train_test_split

PROC_FREQ = 25.  # Hz
SAMP_TIME = 5.0  # Seconds


def readFile(fname):
    """ Reads accelerometer data from file fname into a multi column numpy
    array - one row per data file row.
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
    #print(classArr)
    dataArr = rawArr[:,1:].astype(np.float)
    #print(dataArr)
    #exit(-1)
    return classArr, dataArr




if (__name__ == "__main__"):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inDir", required=True,
	help="path to directory of input data files - one file per category"
        "with one row per 5 second training data sample")
    ap.add_argument("-o", "--outDir", required=False,
	            help="Output directory for preprocessed data",
                    default="outDir_train")

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

        
        if (args['plot']):
            print("plot")
            xRaw = np.linspace(0,
                               dataArr.shape[1]/PROC_FREQ,
                               dataArr.shape[1])
            plt.plot(xRaw,dataArr[0])
            plt.show()

                
    X_train, X_test, Y_train, Y_test = train_test_split(
        classArr, dataArr, test_size=0.33, random_state=42)        

    print("X",X_train.shape, X_test.shape)
    print("Y",Y_train.shape, Y_test.shape)
            
