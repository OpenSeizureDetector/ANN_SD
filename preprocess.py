#!/usr/bin/python

# Preprocess HMP dataset to 25 Hz sample rate (from the 32 Hz provided)
# for compatibility with Garmin watch data.

import numpy as np
import numpy.linalg as la
import scipy.signal
import os
import matplotlib.pyplot as plt

RAW_FREQ = 32.  # Hz
PROC_FREQ = 25.  # Hz

def hmpValTomg(hmpVal):
    """
    Convert a string value in the HMP dataset to a float value in mg,
    using the hmp defined formula:
    The conversion rule to extract the real acceleration value from the coded 
    value is the following:
	real_val = -1.5g + (coded_val/63)*3g
    """
    mgVal = 1000.*(-1.5 + (float(hmpVal)/63.)*3.)
    return (mgVal)

def readFile(fname):
    """ Reads accelerometer data from file fname into a 3 column numpy
    array.
    It  converts the values to mg using the specified formula:
    The conversion rule to extract the real acceleration value from the coded 
    value is the following:
	real_val = -1.5g + (coded_val/63)*3g
    returns a numpy array with one column per axis (x,y,z) and one row per 
    reading at 25Hz sample rate.
    """
    if (os.path.exists(fname)):
        rawArr = np.loadtxt(fname,
                            converters = {
                                0 : hmpValTomg,
                                1 : hmpValTomg,
                                2 : hmpValTomg
                            }
        )
        # print(rawArr)
        rawArr = getMagnitude(rawArr)
    else:
        print("readFile: ERROR - %s does not exist" % fname)
        rawArr = None

    return rawArr


def reSample(inArr, inFreq = 32, outFreq=25):
    inLen = inArr.shape[0]
    outLen = int(inLen * outFreq / inFreq)
    outArr = scipy.signal.resample(inArr,outLen)
    return outArr

def getMagnitude(inArr):
    outArr = la.norm(inArr,axis=1)
    return(outArr)

if (__name__ == "__main__"):
    fname = "HMP_Dataset/Walk/Accelerometer-2012-06-11-11-39-29-walk-m1.txt"
    print("Reading file %s" % fname)
    rawArr = readFile(fname)
    xRaw = np.linspace(0, rawArr.shape[0]/RAW_FREQ, rawArr.shape[0])
    print(rawArr, rawArr.shape)

    print("Num Samples = % d,  duration = %f sec" % (rawArr.shape[0], rawArr.shape[0]/RAW_FREQ))

    procArr = reSample(rawArr,RAW_FREQ,PROC_FREQ)
    xProc = np.linspace(0, procArr.shape[0]/PROC_FREQ, procArr.shape[0])

    print(procArr, procArr.shape)

    plt.plot(xRaw,rawArr)
    plt.plot(xProc,procArr)
    plt.show()

