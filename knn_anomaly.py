#!/usr/bin/env python

# Use real OpenSeizureDetector Data as the 'normal' input to a
# k-Nearest-Neighbours (KNN) based anomaly detector.

import os
import argparse
import numpy as np
# import numpy.linalg as la
# import scipy.signal
import sklearn.decomposition
import sklearn.discriminant_analysis
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import sys
import platform

import osdDataUtils

print("Running Python%s - V %s" %
      (sys.version_info[0], platform.python_version()))
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")



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
    ap.add_argument("-d", "--dims", required=False, default=2,
                    help="Number of dimensions for PCA analysis (2 or 3)")
    ap.add_argument("-m", "--model", required=False, default="PCA",
                    help="Model to be used for visualisation - default is 'PCA'")
    args = vars(ap.parse_args())

    print(args)
    fname = args['inFile']
    nSamp = int(args['nSamp'])
    nDims = int(args['dims'])

    if (nDims != 2 and nDims != 3):
        print("ERROR:  dims must be either 2 or 3")
        exit(-1)

    firstFile = True
    if (os.path.isdir(fname)):
        print("inFile is a directory - reading all files in the folder")
        for root, dirs, files in os.walk(fname):
            print("walk: ", root, dirs, files)
            for inFname in files:
                fn, ext = os.path.splitext(inFname)
                if (ext == ".csv"):
                    print("processing %s in directory %s" % (inFname, root))
                    tArr, hArr, aArr, fArr, qArr = osdDataUtils.processOsdFile(
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
            = osdDataUtils.processOsdFile(fname, nSamp)
    print("accArr", accArr.shape, accArr.dtype, accArr[0, 0].dtype)
    print("fftArr", fftArr.shape)
    print("fftFreq", fftFreq, fftFreq.shape)

    if (args['testFile']is not None):
        testTimeArr, testHrArr, testAccArr, testFftArr, testFftFreq \
            = osdDataUtils.processOsdFile(args['testFile'], nSamp)


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
    #plt.show()

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
    #plt.show()

    # Now collapse all the data down into three dimensions (Rather than 10)
    # so we can plot it and see what it looks like.

    if (nDims==3):
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

        # X_combined = np.absolute(fftArr)
        # X_combined = np.vstack((X_combined,np.absolute(testFftArr)))
        # print("X_combined=",X_combined.shape)
        # Y_combined = np.zeros(X_combined.shape[0])
        # Y_combined[X_reduced.shape[0]+1:] = 1
        # print("Y_combined=",Y_combined.shape,Y_combined)
                              
        # lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=3)
        # lda.fit(X_combined,Y_combined)
        # Xr_lda = lda.transform(X_combined)
        # print("Xr_lda=",Xr_lda.shape,Xr_lda)

        # fig = plt.figure(1, figsize=(8, 6))
        # ax = Axes3D(fig, elev=-120, azim=140)
        # ax.scatter(Xr_lda[:, 0], Xr_lda[:, 1], Xr_lda[:, 2], c="blue",
        #            cmap=plt.cm.Set1, edgecolor='blue', s=1)
        # ax.set_title("First three PCA directions")
        # ax.set_xlabel("1st eigenvector")
        # ax.w_xaxis.set_ticklabels([])
        # ax.set_ylabel("2nd eigenvector")
        # ax.w_yaxis.set_ticklabels([])
        # ax.set_zlabel("3rd eigenvector")
        # ax.w_zaxis.set_ticklabels([])
        
    else:
        # Plot 2d PCA
        fig, ax = plt.subplots(1,2)
        pca = sklearn.decomposition.PCA(n_components=2)
        X_reduced = pca.fit_transform(np.absolute(fftArr))
        X_test = pca.transform(np.absolute(testFftArr))
        print("Plotting normal data points ", X_reduced.shape)
        print("Plotting test data points", X_test.shape)
        ax[0].scatter(X_reduced[:, 0], X_reduced[:, 1], c="blue",
                   cmap=plt.cm.Set1, edgecolor='blue', s=1)
        ax[0].scatter(X_test[:, 0], X_test[:, 1], c="red",
                   cmap=plt.cm.Set1, edgecolor='red', s=20)
        ax[0].set_title("PCA: First two PCA directions")

        # Now do LDA Calculation
        # X_combined = np.absolute(fftArr)
        # X_combined = np.vstack((X_combined,np.absolute(testFftArr)))
        # print("X_combined=",X_combined.shape)
        # Y_combined = np.zeros(X_combined.shape[0])
        # Y_combined[X_reduced.shape[0]+1:] = 1
        # print("Y_combined=",Y_combined.shape,Y_combined)
                              
        # lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
        # lda.fit(X_combined,Y_combined)
        # Xr_lda = lda.transform(X_combined)
        # print("Xr_lda=",Xr_lda.shape,Xr_lda)
        # ax[1].scatter(Xr_lda[:, 0], Xr_lda[:, 1], c="blue",
        #            cmap=plt.cm.Set1, edgecolor='blue', s=1)
        # ax[1].set_title("LDA")

        kpca = sklearn.decomposition.KernelPCA(n_components=2,
                                               kernel='cosine')
        Xkpca_fit = kpca.fit_transform(np.absolute(fftArr))
        Xkpca_test = kpca.transform(np.absolute(testFftArr))
        print("kPCA: Plotting normal data points ", Xkpca_fit.shape)
        print("kPCA: Plotting test data points", Xkpca_test.shape)
        ax[1].scatter(Xkpca_fit[:, 0], Xkpca_fit[:, 1], c="blue",
                   cmap=plt.cm.Set1, edgecolor='blue', s=1)
        ax[1].scatter(Xkpca_test[:, 0], Xkpca_test[:, 1], c="red",
                   cmap=plt.cm.Set1, edgecolor='red', s=20)
        ax[1].set_title("KernelPCA: First two KernelPCA directions")
        
        
    plt.show()
