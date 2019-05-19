#!/usr/bin/env python

# Use real OpenSeizureDetector Data as the 'normal' input to a
# k-Nearest-Neighbours (KNN) based anomaly detector.

import os
import argparse
import numpy as np
# import numpy.linalg as la
# import scipy.signal
import matplotlib.pyplot as plt

import sys
import platform
print("Running Python%s - V %s" %
      (sys.version_info[0], platform.python_version()))
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")



SAMPLE_FREQ = 25.  # Hz

def getData(fHandle):
    """ Reads one line of accelerometer data from the open file
    fHandle.   Returns an array of accelerometer data and a string
    of the time of the data.
    @returns timeStr, accArr
    """
    accStartCol = 16
    accNumSamp = 125
    hrCol = 15

    accArr = []
    timeStr = ""
    hr = -1
    lineStr = fHandle.readline()
    if (lineStr != ""):
        lParts = lineStr.split(",")
        timeStr = lParts[0]
        hr = float(lParts[hrCol])
        for i in range(accStartCol, accStartCol+accNumSamp):
            accArr.append(float(lParts[i]))
        # print("accArr: ",accArr)
    return timeStr, hr, accArr


def buildRawArr(fname, nSamp = 125):
    """ makes a raw input data array from the OpenSeizureDetector
    data log file fname.   
    The output array is one row per sample, and each row is a list
    consisting of the time, heart rate, 
    followed by nSamp data points as read from the file.
    """
    if (os.path.exists(fname)):
        fHandle = open(fname, "r")
        tStr = "start"
        dataArr = []
        rawArr = []
        while (tStr != ""):  # Detect end of file
            # Read 1 row of data from teh input file and add it onto the
            # end of our data Array.
            tStr, hr, dArr = getData(fHandle)
            # print(tStr, dArr)
            dataArr.extend(dArr)
            # print(tStr,len(dataArr))
            # Now we loop through the dataArray and extract samples
            # of nSamp data points.
            # Loop until we have ran out of data in dataArr and need
            # to read some more from the input file.
            while(len(dataArr) >= nSamp):
                sampArr = []
                sampArr.append(tStr)
                sampArr.append(hr)
                # print("len(samp)=%d, len(dataArr)=%d"
                #     % (len(sampArr),len(dataArr)))
                sampArr.extend(dataArr[:nSamp])
                dataArr = dataArr[nSamp:]
                # print("len(samp)=%d, len(dataArr)=%d"
                #       % (len(sampArr),len(dataArr)))
                rawArr.append(sampArr)

        print("End of File")
        return(rawArr)

    else:
        print("readFile: ERROR - %s does not exist" % fname)
        exit(-1)



if (__name__ == "__main__"):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inFile", required=True,
                    help="input data file")
    ap.add_argument("-p", "--plot", dest='plot', action='store_true',
                    help="Plot the data as it is processed")
    args = vars(ap.parse_args())

    print(args)
    fname = args['inFile']

    rawArr = buildRawArr(fname, nSamp=25)
    rawArr = np.asarray(rawArr)

    print(rawArr, rawArr.shape, rawArr.dtype, rawArr[0, 0].dtype)

    # Calculate the FFT of all of the 
    fftArr = np.fft.fft(rawArr[:, 2:])
    fftFreq = np.fft.fftfreq(fftArr.shape[-1], 1.0/SAMPLE_FREQ)
    fftLen = int(fftArr.shape[-1]/2)

    print(fftArr, fftArr.shape)
    print(fftFreq, fftFreq.shape)
    print(fftLen)

    n = 0
    for row in rawArr:
        print(n, row)
        n += 1

    recNo = 22164
    print(rawArr[recNo, 2:2].dtype)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(rawArr[recNo, 2:])
    
    # Plot fft, but chop off large DC term, and the negative frequency part.
    ax[1].plot(fftFreq[1:fftLen],
               np.absolute(fftArr[recNo, 1:fftLen]))
    plt.show()



