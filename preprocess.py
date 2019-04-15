#!/usr/bin/python

# Preprocess HMP dataset to 25 Hz sample rate (from the 32 Hz provided)
# for compatibility with Garmin watch data.

import os
import argparse
import numpy as np
import numpy.linalg as la
import scipy.signal
import matplotlib.pyplot as plt
from imutils import paths

RAW_FREQ = 32.  # Hz
PROC_FREQ = 25.  # Hz
SAMP_TIME = 5.0  # Seconds

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

def readFile(fname, trim=0):
    """ Reads accelerometer data from file fname into a 3 column numpy
    array.
    It  converts the values to mg using the specified formula:
    The conversion rule to extract the real acceleration value from the coded 
    value is the following:
	real_val = -1.5g + (coded_val/63)*3g
    trims 'trim' seconds of data off each end of the dataset to get 
    rid of edge effects.
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

    nTrim = int(trim * RAW_FREQ)
    if (rawArr.shape[0]<=(2*nTrim)):
        print("Dataset too short to trim - using full dataset")
    else:
        # print ("Trim=%.1f, nTrim=%d" % (trim,nTrim))
        # print(rawArr.shape)
        rawArr = rawArr[nTrim:rawArr.shape[0]-nTrim]
        # print(rawArr.shape)
    rawArr = reSample(rawArr,RAW_FREQ,PROC_FREQ)
    return rawArr


def reSample(inArr, inFreq = 32, outFreq=25):
    inLen = inArr.shape[0]
    outLen = int(inLen * outFreq / inFreq)
    outArr = scipy.signal.resample(inArr,outLen)
    return outArr

def getMagnitude(inArr):
    outArr = la.norm(inArr,axis=1)
    return(outArr)

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
            

print("preprocess body")
    
if (__name__ == "__main__"):
    print("preprocess.main")
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inDir", required=True,
	help="path to directory of input data files - should be a tree structure with the final folder name being the class of the activity represented by the data")
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

    fileList = list(paths.list_files(args['inDir'],validExts=('.txt')))

    #print(fileList)
                                
    for fname in fileList:
        fnameParts = fname.split("/")
        className = fnameParts[len(fnameParts)-2]
        print("Reading file %s" % fname)
        print("className=%s" % className)
        rawArr = readFile(fname, float(args['trim']))
        xRaw = np.linspace(0, rawArr.shape[0]/PROC_FREQ, rawArr.shape[0])
        dur = rawArr.shape[0]/PROC_FREQ
        print("Num Samples = % d,  duration = %.1f sec" % (rawArr.shape[0], dur))
        if (dur<5.):
            print("Skipping short duration sample")
        else:
            if (args['plot']):
                plt.plot(xRaw,rawArr)
                plt.show()
            writeData(className, rawArr, args['outDir'])
                
            
