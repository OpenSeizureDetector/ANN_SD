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
