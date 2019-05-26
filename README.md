ANN_SD - An Artificial Neural Network Based Seizure Detector
============================================================

Note:  This is in a very early stage of development - we are learning
about training neural networks rather than turning it into a practical 
seizure detector at the moment.

Objective
=========
The objective of this project is to develop an alternative seizure detection
algorithm for OpenSeizureDetector.   The algorithm that is currently in use
is Deterministic - it assumes that a high enough spectral power within
a given region of interest is indicative of seizure like movement.   This means
it is prone to false alarms for activities that have movements in a similar
frequency range, such as brushing teeth.
The hope is that by adopting a Neural Network based detection system that
we will be able to teach it more subtle differences between a seizure and
these other activities.
Success is far from guaranteed!

The Plan
========
Making a neural network is pretty easy - will use the Python Keras library at
least for development.
The problem is that to train it you need lots of data, which is why I haven't 
tried to do it before.   For this reason we will use a 2 stage approach to
development, with the first stage using mostly published data to prove that
we can detect specific activities from accelerometer data.   Hopefully this
will allow us to produce a fall detector, but it will not detect seizures.
The second stage will need us to collect data from volunteer users to help us
collect enough seizure-like data to train it to detect that.

* Stage 1
  *  Collect as much published, categorised accelerometer data as we can 
  together, and process it into a single dataset that is sampled at the same
  rate as we get from the OpenSeizureDetector watches..
  *  Train a Neural Network Classifier with the data and check that we can 
  detect the different classes of activity using it.
  *  If the available data does not include falls, simulate many falls
  and add the data from them to the dataset.
  *  Check that we can detect falls reliably.
  
* Stage 2
  *  Collect data from users, including real seizures, and convert it into a
  classified dataset.
  *  Extend the neural network to include the seizure data.
  *  Tweak the neural network number of layers and size of layers to optimise
  reliability.


Conventional Machine Learning Approach
======================================
Before we get bogged down in Neural Networks, Iain Campbell has helpfully
pointed out that we may be able to do it with a more conventional
machine learning approach such as K-Nearest-Neighbours (KNN).
So we are having a look at that first - it is a useful start anyway
to get us visualising the data so we know what we are working with, rather
than treating a neural network as a black box.....

So the first approach is to see if we can distinguish between seizure like
movements and normal data.  To do this we use a similar analysis approach
to the current OpenSeizureDetection algorithm - calculate the fourier transform
of the accelerometer data to give a frequency spectrum, and put it into
1Hz bins so we have about 10 dimenssions 0-1Hz, 1-2Hz.....9-10Hz.

Unfortuntely it is difficult to visualise things in 10 dimensional space,
so we use a technique called PCA (Principal Component Analysis) to collapse
the 10 dimensions down into 3, which we can plot.

We then plot a backgorund of small dots for each sample of 'normal' data,
and then larger contrasting dots for the seizure data (we are actually using
simulated seizure like movements, because real seizure data is very sparse).
The results are shown in TestData/Results/PCA_1.png.   These show that the
seizure data is on the very edge of the normal data, which is encouraging,
but it does not form a separate population, which is a shame because that
would have made it easier to detect seizures without false alarms.....but
I think we knew that it was difficult!  We had similar results when we
removed the zero offset (gravity) from the seizure data too.
![PCA_1.png - Results from PCA analysis of normal and simulated seizures](https://raw.githubusercontent.com/OpenSeizureDetector/ANN_SD/master/TestData/Results/PCA_1.png)

LDA
===
I have had a look at using LDA (Linear Determinant Analysis), but this seems
to only be a valid way of dimension reduction in a multi-class situation.
At best we only have two classes 'normal' and 'seizure' - I couldn't get it
to make a 2d plot for me.

KernelPCA
=========
KernelPCA is a non-linear version of PCA, so try that next.
