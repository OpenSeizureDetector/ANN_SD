#!/usr/bin/env python


  # osdDataUtils - Utility functions for reading OpenSeizureDetector
  # data log files and pre-processing them for use in machine learning
  # applications.
  #
  # See http://openseizuredetector.org for more information.
  #
  # Copyright Graham Jones, 2019.
  #
  # This file is part of ANN_SD.
  #
  # ANN_SD is free software: you can redistribute it and/or modify
  # it under the terms of the GNU General Public License as published by
  # the Free Software Foundation, either version 3 of the License, or
  # (at your option) any later version.
  #
  # ANN_SD is distributed in the hope that it will be useful,
  # but WITHOUT ANY WARRANTY; without even the implied warranty of
  # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  # GNU General Public License for more details.
  #
  # You should have received a copy of the GNU General Public License
  # along with ANN_SD.  If not, see <http://www.gnu.org/licenses/>.


# osdDataUtils.py
# This module contains a number of utility functions to read
# OpenSeizureDetector data, re-sample to a given sample frequency
# and collect it into samples of given length.
#
import os
import numpy as np

SAMPLE_FREQ = 25.  # Hz


def getDataPoint(fHandle):
    """ Reads one line of accelerometer data from the open file
    fHandle.   Returns an array of accelerometer data and a string
    of the time of the data.
    @returns timeStr, accArr
    """
    alarmStateCol = 14
    hrCol = 15
    accStartCol = 16
    accNumSamp = 125

    accArr = []
    timeStr = ""
    alarmStateStr = ""
    hr = -1
    lineStr = fHandle.readline()
    if (lineStr != ""):
        lParts = lineStr.split(",")
        timeStr = lParts[0]
        alarmStateStr = lParts[alarmStateCol]
        hr = float(lParts[hrCol])
        for i in range(accStartCol, accStartCol+accNumSamp):
            accArr.append(float(lParts[i]))
        # print("accArr: ",accArr)
    return timeStr, alarmStateStr, hr, accArr


def readOsdFile(fname, nSamp=125):
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
            tStr, alarmStateStr, hr, dArr = getDataPoint(fHandle)
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
                sampArr.append(alarmStateStr)
                sampArr.append(hr)
                # print("len(samp)=%d, len(dataArr)=%d"
                #     % (len(sampArr),len(dataArr)))
                sampArr.extend(dataArr[:nSamp])
                dataArr = dataArr[nSamp:]
                # print("len(samp)=%d, len(dataArr)=%d"
                #       % (len(sampArr),len(dataArr)))
                rawArr.append(sampArr)

        #print("End of File")
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
    rawArr = readOsdFile(fname, nSamp=nSamp)
    rawArr = np.asarray(rawArr)

    #print("rawArr", rawArr, rawArr.shape, rawArr.dtype, rawArr[0, 0].dtype)

    accArr = rawArr[:, 3:].astype(np.float)
    # Remove the DC offset from the acceleration readings
    rowMeans = accArr.mean(axis=1)
    #print("rowMeans=",rowMeans)
    rowMeans = rowMeans.reshape((accArr.shape[0],1))
    accArr = accArr - rowMeans
    #print("rowMeans ", accArr.mean(axis=1))
    hrArr = rawArr[:, 2].astype(np.float)
    timeArr = rawArr[:, 0]
    alarmStateArr = rawArr[:,1]

    # Calculate the FFT of all of the acceleration samples
    fftArr = np.fft.fft(accArr)
    fftFreq = np.fft.fftfreq(fftArr.shape[-1], 1.0/SAMPLE_FREQ)
    fftLen = int(fftArr.shape[-1]/2)
    fftArr = fftArr[:, 1:fftLen]
    fftFreq = fftFreq[1:fftLen]

    return timeArr, alarmStateArr, hrArr, accArr, fftArr, fftFreq


def getOsdData(fname, nSamp):
    firstFile = True
    if (not os.path.exists(fname)):
        print("ERROR: %s does not exist" % fname)
        exit(-1)
    else:
        print("getOsdData - %s exists" % fname)
        
    if (os.path.isdir(fname)):
        print("%s is a directory - reading all files in the folder" % fname)
        for root, dirs, files in os.walk(fname):
            print("walk: ", root, dirs, files)
            for inFname in files:
                fn, ext = os.path.splitext(inFname)
                if (ext == ".csv"):
                    print("processing %s in directory %s" % (inFname, root))
                    tArr, almArr, hArr, aArr, fArr, qArr = processOsdFile(
                        os.path.join(root, inFname), nSamp)
                    if (firstFile):
                        timeArr = tArr
                        alarmStateArr = almArr
                        hrArr = hArr
                        accArr = aArr
                        fftArr = fArr
                        fftFreq = qArr
                        firstFile = False
                    else:
                        #print("timeArr", timeArr.shape, "tArr", tArr.shape)
                        timeArr = np.concatenate((timeArr, tArr))
                        alarmStateArr = np.concatenate((alarmStateArr, almArr))
                        #print("timeArr", timeArr.shape, "tArr", tArr.shape)
                        hrArr = np.concatenate((hrArr,   hArr))
                        #print("accArr", accArr.shape, "aArr", aArr.shape)
                        accArr = np.vstack((accArr,  aArr))
                        #print("accArr", accArr.shape, "aArr", aArr.shape)
                        #print("fftArr", fftArr.shape, "fArr", fArr.shape)
                        fftArr = np.vstack((fftArr,  fArr))
                        #print("fftArr", fftArr.shape, "fArr", fArr.shape)
    else:
        timeArr, alarmStateArr, hrArr, accArr, fftArr, fftFreq \
            = processOsdFile(fname, nSamp)
    # print("accArr", accArr.shape, accArr.dtype, accArr[0, 0].dtype)
    # print("fftArr", fftArr.shape)
    # print("fftFreq", fftFreq, fftFreq.shape)

    return timeArr, alarmStateArr, hrArr, accArr, fftArr, fftFreq


if (__name__ == "__main__"):
    print("osdDataUtils");
    timeArr, alarmStateArr, hrArr, \
        accArr, fftArr, fftFreq = getOsdData("TestData/Normal",25)

    print("accArr", accArr.shape, accArr.dtype, accArr[0, 0].dtype)
    print("fftArr", fftArr.shape)
    print("fftFreq", fftFreq, fftFreq.shape)

    
