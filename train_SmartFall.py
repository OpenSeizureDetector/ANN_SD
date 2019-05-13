#!/usr/bin/env python

# Train a dense neural network with accelerometer data.

import os
import argparse
import numpy as np
import numpy.linalg as la
import scipy.signal
import matplotlib.pyplot as plt
import keras.utils
import keras.models
import keras.layers

import sklearn
print(sklearn.__version__)
from sklearn.model_selection import train_test_split
import sklearn.preprocessing

import sys
import platform

import accUtil

print("Running Python%s - V %s" % (sys.version_info[0],platform.python_version()))
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")


PROC_FREQ = 25.  # Hz
SAMP_TIME = 5.0  # Seconds

HIDDEN_UNITS = 32
DROPOUT = 0.45
BATCH_SIZE = 128
EPOCHS = 100

def readFile(fname):
    """ Reads accelerometer data from file fname into a multi column numpy
    array - one row per data file row.
    """
    if (os.path.exists(fname)):
        # rawArr = np.genfromtxt(fname,delimiter=",", dtype=None)
        rawArr = np.loadtxt(fname,delimiter=",", dtype=np.str)
    else:
        print("readFile: ERROR - %s does not exist" % fname)
        rawArr = None

    classArr = rawArr[:,0]
    dataArr = rawArr[:,1:].astype(np.float)

    # print(rawArr, rawArr.shape)
    # print(rawArr.shape)
    # print(classArr)
    # print(dataArr)
    # exit(-1)
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

    print(args)

    fileList = []
    for r, d, f in os.walk(args['inDir']):
        for fname in f:
            if ('.csv' in fname):
                fileList.append(os.path.join(r,fname))

    if (len(fileList)==0):
        print("***ERROR - No input files found ****")
        exit(-1)
        
    print(fileList)

    classArr = None
    dataArr = None
    
    for fname in fileList:
        fnameParts = fname.split("/")
        className = fnameParts[len(fnameParts)-1].split(".")[0]
        print("Reading file %s" % fname)
        print("className=%s" % className)
        cArr, dArr = readFile(fname)

        # print(cArr, cArr.shape)
        # print(dArr, dArr.shape)
        
        if (classArr is None):
            classArr = cArr
        else:
            print("classArr",classArr.shape, cArr.shape)
            classArr = np.hstack((classArr, cArr))
            print("classArr.shape",classArr.shape)

        if (dataArr is None):
            dataArr = dArr
        else:
            print("dataArr",dataArr.shape, dArr.shape)
            dataArr = np.vstack((dataArr, dArr))
            print("dataArr.shape",dataArr.shape)

        # print(classArr.shape, dataArr.shape)


    # Normalise the data
    print("Before Normalisation DataArr mean=%f, max=%f, min=%f" %
          (dataArr.mean(),dataArr.max(),dataArr.min()))
    dataArr = accUtil.normaliseAccArr(dataArr)
    print("After  Normalisation DataArr mean=%f, max=%f, min=%f" %
          (dataArr.mean(),dataArr.max(),dataArr.min()))
                
    # Split it into training and testing datasets
    label_train, label_test, data_train, data_test = train_test_split(
        classArr, dataArr, test_size=0.33, random_state=42)        

    print("Labels",label_train.shape, label_test.shape)
    print("Data",data_train.shape, data_test.shape)
            
    # count the number of unique train labels
    unique, counts = np.unique(label_train, return_counts=True)
    print("Unique:",unique," Counts:",counts)
    print("Train labels: ", dict(zip(unique, counts)))
    # count the number of unique test labels
    unique, counts = np.unique(label_test, return_counts=True)
    print("Test labels: ", dict(zip(unique, counts)))

    encoder = sklearn.preprocessing.LabelBinarizer()
    label_train_bin = encoder.fit_transform(label_train)
    label_test_bin = encoder.fit_transform(label_test)

    print(label_train_bin)
    print(encoder.classes_)

    label_train_cat = keras.utils.to_categorical(label_train_bin,num_classes=2)
    label_test_cat = keras.utils.to_categorical(label_test_bin,num_classes=2)
    #print(dir(label_train_bin))
    # exit(-1)

    if (args['plot']):
        print("Plotting Sample of data")
        # Plot 25 random datasets
        indices = np.random.randint(0, data_train.shape[0], size=25)
        yData = data_train[indices]
        labels = label_train[indices]
        xVals = np.linspace(0,
                           data_train.shape[1]/PROC_FREQ,
                           data_train.shape[1])
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
        print("[INFO] creating model...data_train.shape=",data_train.shape)
        print("Data Train: ",data_train)
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(HIDDEN_UNITS, input_dim=data_train.shape[1]))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(DROPOUT))
        model.add(keras.layers.Dense(HIDDEN_UNITS))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(DROPOUT))
        model.add(keras.layers.Dense(len(unique)))
        model.add(keras.layers.Activation('softmax'))

    model.summary()
    keras.utils.plot_model(model, to_file='model.png', show_shapes = True)

    filepath="acc_ann-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    H = model.fit(data_train, label_train_cat,
              validation_data=(data_test, label_test_cat),
              epochs = EPOCHS, batch_size=BATCH_SIZE,
              callbacks = callbacks_list
    )

    # save the model to disk
    print("[INFO] serializing network...")
    model.save(args["model"])
    
    loss, acc = model.evaluate(data_test, label_test_cat, batch_size = BATCH_SIZE)
    print("\nTest accuracy: %0.1f%%" % (100. * acc))

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy for Farm identification")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("training.png")

