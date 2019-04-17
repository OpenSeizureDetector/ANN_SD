#!/usr/bin/python3

# Preprocess SmartFall dataset to 25 Hz sample rate (from the 31.25 Hz provided)
# for compatibility with Garmin watch data.

import os
import argparse
import numpy as np
import numpy.linalg as la
import scipy.signal
import matplotlib.pyplot as plt
#from imutils import paths

RAW_FREQ = 31.25  # Hz
PROC_FREQ = 25.  # Hz
SAMP_TIME = 1.5  # Seconds

def getMagnitude(inArr):
    """ Return an array that is the magnitude of the three column inArr
    """
    outArr = la.norm(inArr,axis=1)
    return(outArr)

def rawTomg(hmpVal):
    """
    Convert a string value in the SmartFall dataset to a float value in mg,
    Very simple, because Smartfal numbers are g
    """
    mgVal = 1000.*float(hmpVal)
    return (mgVal)

def readFile(fname, trim=0):
    """ Reads accelerometer data from file fname into a 4 column numpy
    array.
    It  converts the values to mg using the rawTomg function:
    trims 'trim' seconds of data off each end of the dataset to get 
    rid of edge effects.
    returns a numpy array with one column per axis (x,y,z,f) and one row per 
    reading at 25Hz sample rate.   f is a flag to signify a fall.
    """
    if (os.path.exists(fname)):
        print("readFile - %s Exists - reading it" % fname)
        rawArr = np.loadtxt(fname,
                            delimiter = ',',
                            converters = {
                                0 : rawTomg,
                                1 : rawTomg,
                                2 : rawTomg,
                                3 : int
                            },
                            skiprows = 1  # Skip the first (header) row
        )
        #print("rawArr = ",rawArr)
        # Split the class identifier column from the 3 acceleration columns
        classArr = rawArr[:,3]
        accArr = rawArr[:,[0,1,2]]
        # Calculate the magnitude of the acceleration vectors
        magArr = getMagnitude(accArr)
        # And combine the acceleration magnitude and class id back into a 2d array.
        magArr = np.stack((magArr,classArr),axis=-1)
        # print("classArr=",classArr,classArr.shape)
        # print("accArr=",accArr,accArr.shape)
        # print("magArr = ",magArr, magArr.shape)

        

        # Now split the array into separate arrays where the class column (1)
        # Changes.
        # See https://stackoverflow.com/questions/31863083/python-split-numpy-array-based-on-values-in-the-array
        # Basically
        #    arr[:,1] selects all the column 1 values.
        #    np.diff calculates the difference between consecutive values.
        #    np.where selects non-zero differences.....
        # >>> np.split(arr, np.where(np.diff(arr[:,1]))[0]+1)

        splitArr = np.split(magArr,np.where(np.diff(magArr[:,1]))[0]+1)
        # This gives us splitArr, which is a list of np arrays
        # print("splitArr=",len(splitArr),splitArr[0],splitArr[0].shape,splitArr[1].shape)

    else:
        print("readFile: ERROR - %s does not exist" % fname)
        rawArr = None
        exit(-1)

    nTrim = int(trim * RAW_FREQ)
    procArrs = []
    for eventArr in splitArr:
        if (eventArr.shape[0]<=(2*nTrim)):
            print("Dataset too short to trim - using full dataset")
        else:
            # print ("Trim=%.1f, nTrim=%d" % (trim,nTrim))
            # print(eventArr.shape)
            eventArr = eventArr[nTrim:eventArr.shape[0]-nTrim]
            # print(eventArr.shape)
        reSampArr = reSample(eventArr,RAW_FREQ,PROC_FREQ)
        print("Re-sampled from %d points to %d points" %
              (eventArr.shape[0], reSampArr.shape[0]))
        procArrs.append(reSampArr)
    # print("procArrs = ",len(procArrs),procArrs[0],procArrs[0].shape)
    return procArrs


def reSample(inArr, inFreq = 32, outFreq=25):
    inLen = inArr.shape[0]
    outLen = int(inLen * outFreq / inFreq)
    outArr = scipy.signal.resample(inArr,outLen)
    return outArr


def writeData(catName, dataArr, outDir, dataStep = 3):
    """ Appends the data in dataArr to output CSV file <outDir>/<catName>.txt
    Each SAMP_TIME period of data is written on a separate line.
    The start position in dataArr is incremented by dataStep and a new line
    written until the end of dataArr is reached.
    """
    if (not os.path.exists(outDir)):
        os.makedirs(outDir)
    outFname = os.path.join(outDir, "%s.csv" % catName)

    nSamp = int(SAMP_TIME * PROC_FREQ)
    f = open(outFname,"a")

    startPos = 0

    while ((startPos + nSamp) < len(dataArr)):
        print("Writing from startPos %d" % startPos)
        outStr = ""
        for n in range(0,nSamp):
            if (n>0):
                outStr = "%s, %0.1f" % (outStr,dataArr[startPos+n])
            else:
                outStr = "%s, %0.1f" % (catName,dataArr[startPos+n])
        f.write(outStr)
        f.write("\n")
        startPos += dataStep
    f.close()
            

if (__name__ == "__main__"):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inFile", required=True,
	help="path to the smartFall data file to use")
    ap.add_argument("-o", "--outDir", required=False,
	            help="Output directory for preprocessed data",
                    default="outDir")
    ap.add_argument("-t", "--trim", required=False,
	            help="Number of seconds to trim off each dataset to get rid of edge effects",
                    default=0)

    ap.add_argument("-p", "--plot", dest='plot',action='store_true',
	            help="Plot the data as it is processed")
    args = vars(ap.parse_args())

    print(args)

    fname = args['inFile']

    fnameParts = fname.split("/")
    print("Reading file %s" % fname)
    procArrs = readFile(fname, float(args['trim']))

    for eventArr in procArrs:
        if (eventArr[0][1]):
            print("Found Fall Event")
            className = "Fall"
        else:
            print("Found Normal Activity Event")
            className = "Normal"
        xVals = np.linspace(0, eventArr.shape[0]/RAW_FREQ, eventArr.shape[0])
        if (args['plot']):
            plt.plot(xVals, eventArr, 'r')
            plt.show()
        dur = eventArr.shape[0]/PROC_FREQ
        print("Num Samples = % d,  duration = %.1f sec" % (eventArr.shape[0], dur))
        if (args['plot']):
            plt.plot(xRaw,rawArr)
            plt.show()
        #writeData(className, eventArr, args['outDir'])
                
            
