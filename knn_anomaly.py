#!/usr/bin/env python

# Use real OpenSeizureDetector Data as the 'normal' input to a
# k-Nearest-Neighbours (KNN) based anomaly detector.

import os
import argparse
import numpy as np
# import numpy.linalg as la
# import scipy.signal
import sklearn.decomposition
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
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
    rawArr = buildRawArr(fname, nSamp=25)
    rawArr = np.asarray(rawArr)

    print("rawArr", rawArr, rawArr.shape, rawArr.dtype, rawArr[0, 0].dtype)

    accArr = rawArr[:, 2:].astype(np.float)
    hrArr = rawArr[:, 1].astype(np.float)
    timeArr = rawArr[:, 0]

    # Calculate the FFT of all of the acceleration samples
    fftArr = np.fft.fft(accArr)
    fftFreq = np.fft.fftfreq(fftArr.shape[-1], 1.0/SAMPLE_FREQ)
    fftLen = int(fftArr.shape[-1]/2)
    fftArr = fftArr[:, 1:fftLen]
    fftFreq = fftFreq[1:fftLen]

    return timeArr, hrArr, accArr, fftArr, fftFreq


if (__name__ == "__main__"):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inFile", required=True,
                    help="input data file")
    ap.add_argument("-t", "--testFile", required=False, default=None,
                    help="test data file")
    ap.add_argument("-n", "--nSamp", required=False, default=25,
                    help="numper of data points to use per sample")
    ap.add_argument("-f", "--freq", required=False, default=25.0,
                    help="sample frequency in input data file")
    ap.add_argument("-p", "--plot", dest='plot', action='store_true',
                    help="Plot the data as it is processed")
    args = vars(ap.parse_args())

    print(args)
    fname = args['inFile']
    nSamp = int(args['nSamp'])

    firstFile = True
    if (os.path.isdir(fname)):
        print("inFile is a directory - reading all files in the folder")
        print("FIXME - THIS DOESN'T WORK YET")
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
        timeArr, hrArr, accArr, fftArr, fftFreq = processOsdFile(fname, nSamp)
    print("accArr", accArr.shape, accArr.dtype, accArr[0, 0].dtype)
    print("fftArr", fftArr.shape)
    print("fftFreq", fftFreq, fftFreq.shape)

    if (args['testFile']is not None):
        testTimeArr, testHrArr, testAccArr, testFftArr, testFftFreq \
            = processOsdFile(args['testFile'], nSamp)


    # n = 0
    # for row in accArr:
    #     print(n, row)
    #     n += 1

    nPlots = 2
    # Plot nPlots random samples
    fig, ax = plt.subplots(nPlots, 2)
    for n in range(0, nPlots):
        recNo = np.random.randint(0, accArr.shape[0])
        print("Sample Normal Data: Plotting Record Number %d: "
              % recNo, accArr[recNo, :])
        xvals = np.arange(0, accArr.shape[-1])
        ax[n, 0].plot(xvals, accArr[recNo, :])
        ax[n, 0].set_title = "Rcec %d" % recNo

        # Plot fft, but chop off large DC term, and the neg. frequency part.
        ax[n, 1].plot(fftFreq,
                      np.absolute(fftArr[recNo, :]))
    plt.show()

    nPlots = 2
    # Plot nPlots random samples
    fig, ax = plt.subplots(nPlots, 2)
    for n in range(0, nPlots):
        recNo = np.random.randint(0, testAccArr.shape[0])
        print("Plotting test data: Record Number %d: " %
              recNo, testAccArr[recNo, :])
        xvals = np.arange(0, testAccArr.shape[-1])
        ax[n, 0].plot(xvals, testAccArr[recNo, :])
        ax[n, 0].set_title = "Rcec %d" % recNo

        # Plot fft, but chop off large DC term, and the neg. frequency part.
        ax[n, 1].plot(testFftFreq,
                      np.absolute(testFftArr[recNo, :]))
    plt.show()

    # Now collapse all the data down into three dimensions (Rather than 10)
    # so we can plot it and see what it looks like.

    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-120, azim=140)
    pca = sklearn.decomposition.PCA(n_components=3)
    X_reduced = pca.fit_transform(np.absolute(fftArr))
    X_test = pca.transform(np.absolute(testFftArr))
    print("Plotting normal data points ", X_reduced.shape)
    print("Plotting test data points", X_test.shape)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c="blue",
               cmap=plt.cm.Set1, edgecolor='blue', s=1)
    ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c="red",
               cmap=plt.cm.Set1, edgecolor='red', s=40)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    plt.show()
