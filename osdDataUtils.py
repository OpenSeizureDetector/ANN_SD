#!/usr/bin/env python

# osdDataUtils.py
# This module contains a number of utility functions to read
# OpenSeizureDetector data, re-sample to a given sample frequency
# and collect it into samples of given length.
#
import os
import numpy as np

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


def buildRawArr(fname, nSamp=125):
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
        fHandle.close()
        return(rawArr)

    else:
        print("readFile: ERROR - %s does not exist" % fname)
        exit(-1)

def processOsdFile(fname, nSamp):
    """ Processes an OpenSeizureDetector DataLog file, splits it into
    periods of length nSamp samples, and calculates the power spectrum
    of the data for each period.
    It returns 3 numpy arrays:
            timeArr, which is the data times,
            hrArr which is measured heart rate,
            accArr which is the raw acceleration data,
            fftArr which is the spectral power of the accelerometer data.
            fftFreq is the frequencies in fftArr, based on the SAMPLE_FREQ
                    global value.
    """
    print("processOsdFile(%s, %d)" % (fname, nSamp))
    rawArr = buildRawArr(fname, nSamp=nSamp)
    rawArr = np.asarray(rawArr)

    print("rawArr", rawArr, rawArr.shape, rawArr.dtype, rawArr[0, 0].dtype)

    accArr = rawArr[:, 2:].astype(np.float)
    # Remove the DC offset from the acceleration readings
    rowMeans = accArr.mean(axis=1)
    print("rowMeans=",rowMeans)
    rowMeans = rowMeans.reshape((accArr.shape[0],1))
    accArr = accArr - rowMeans
    print("rowMeans ", accArr.mean(axis=1))
    hrArr = rawArr[:, 1].astype(np.float)
    timeArr = rawArr[:, 0]

    # Calculate the FFT of all of the acceleration samples
    fftArr = np.fft.fft(accArr)
    fftFreq = np.fft.fftfreq(fftArr.shape[-1], 1.0/SAMPLE_FREQ)
    fftLen = int(fftArr.shape[-1]/2)
    fftArr = fftArr[:, 1:fftLen]
    fftFreq = fftFreq[1:fftLen]

    return timeArr, hrArr, accArr, fftArr, fftFreq


def getOsdData(fname, nSamp):
    firstFile = True
    if (os.path.isdir(fname)):
        print("inFile is a directory - reading all files in the folder")
        for root, dirs, files in os.walk(fname):
            print("walk: ", root, dirs, files)
            for inFname in files:
                fn, ext = os.path.splitext(inFname)
                if (ext == ".csv"):
                    print("processing %s in directory %s" % (inFname, root))
                    tArr, hArr, aArr, fArr, qArr = processOsdFile(
                        os.path.join(root, inFname), nSamp)
                    if (firstFile):
                        timeArr = tArr
                        hrArr = hArr
                        accArr = aArr
                        fftArr = fArr
                        fftFreq = qArr
                        firstFile = False
                    else:
                        print("timeArr", timeArr.shape, "tArr", tArr.shape)
                        timeArr = np.concatenate((timeArr, tArr))
                        print("timeArr", timeArr.shape, "tArr", tArr.shape)
                        hrArr = np.concatenate((hrArr,   hArr))
                        print("accArr", accArr.shape, "aArr", aArr.shape)
                        accArr = np.vstack((accArr,  aArr))
                        print("accArr", accArr.shape, "aArr", aArr.shape)
                        print("fftArr", fftArr.shape, "fArr", fArr.shape)
                        fftArr = np.vstack((fftArr,  fArr))
                        print("fftArr", fftArr.shape, "fArr", fArr.shape)
    else:
        timeArr, hrArr, accArr, fftArr, fftFreq \
            = processOsdFile(fname, nSamp)
    print("accArr", accArr.shape, accArr.dtype, accArr[0, 0].dtype)
    print("fftArr", fftArr.shape)
    print("fftFreq", fftFreq, fftFreq.shape)

    return timeArr, hrArr, accArr, fftArr, fftFreq
