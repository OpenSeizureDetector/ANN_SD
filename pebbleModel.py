#!/usr/bin/env python

# pebbleModel:  A python implementation of the seizure detection model
# implemented on the OpenSeizureDetector pebble watch.
#
# It analyses 5 seconds of accelerometer data and calculates the spectral
# power distribution.
# The power within a given region of interest (ROI) is compared to the whole
# spectrum power.   If the ratio of the two exceeds a specified threhold,
# the movement in the time period is considered seizure like.
#
# Two consecutive seizure like periods triggers an alarm.

import argparse
import numpy as np
import matplotlib.pyplot as plt

import osdDataUtils


def readConfig(configFname):
    print("readConfig")


defaultConfig = {
    'alarmThreshold' : 100.,
    'alarmRatioThreshold' : 50,
    'nSamp' : 125,
    'sampleFreq' : 25,
    'alarmFreqMin' : 3.0,
    'alarmFreqMax' : 10.0,
    'freqCutoff' : 12.0,
    'warnTime' : 5.0,
    'alarmTime' : 10.0,
}


class PebbleModel:
    config = None
    alarmCount = 0
    alarmState = 0
    
    def __init__(self,config):
        print("PebbleModel.__init__()", config)
        self.config = config

    def doAnalysis(self, accData, plotData = False):
        # print("pebbleModel.doAnalysis()",accData.shape)
        freqRes =  1.0 * config['sampleFreq'] / config['nSamp']
        # print("freqRes = %f Hz" % freqRes)

        nMin = int(config['alarmFreqMin'] / freqRes)
        nMax = int(config['alarmFreqMax'] / freqRes)
        nFreqCutoff = int(config['freqCutoff'] / freqRes)
        # print("nMin,nMax, cutoff = %d, %d, %d" % (nMin, nMax, nFreqCutoff))

        fftArr = np.fft.fft(accData)
        fftFreq = np.fft.fftfreq(fftArr.shape[-1], 1.0/config['sampleFreq'])

        specPower = 0.

        # range starts at 1 to avoid the DC component
        for i in range(1,config['nSamp']/2):
            if (i<=nFreqCutoff):
                specPower = specPower + self.getMagnitude(fftArr[i])
        # The following may not be correct
        specPower = specPower / config['nSamp'] / 2

        roiPower = 0.
        for i in range(nMin,nMax):
            roiPower = roiPower + self.getMagnitude(fftArr[i])
        roiPower = roiPower/(nMax-nMin)

        roiRatio = 10 * roiPower/specPower

        #print("fftArr", fftArr)
        #print("fftFreq",fftFreq)

        #print("specPower=%f, roiPower=%f, roiRatio=%f" %
        #      (specPower, roiPower, roiRatio))


        
        # Now do the alarm checks
        inAlarm = (roiPower > config['alarmThreshold']) and \
                  (roiRatio > config['alarmRatioThreshold'])
        

        if (inAlarm):
            #print("inAlarm - roiPower=%f, roiRatio=%f" % (roiPower, roiRatio))
            self.alarmCount += config['nSamp'] * config['sampleFreq']
            #print("alarmCount=%d" % self.alarmCount)

            if (self.alarmCount > config['alarmTime']):
                self.alarmState = 2
            elif (self.alarmCount > config['warnTime']):
                self.alarmState = 1
        else:
            # if we are not in alarm state revert back to warning or ok.
            if (self.alarmState == 2):
                self.alarmState = 1
                self.alarmCount = config['warnTime'] + 1
            else:
                self.alarmState = 0
                self.alarmCount = 0

        extraData = {
            'specPower': specPower,
            'roiPower': roiPower,
            'roiRatio': roiRatio,
            'alarmCount': self.alarmCount,
            'alarmState': self.alarmState,
            'fftArr': fftArr,
            'fftFreq': fftFreq,
            }
        return self.alarmState, extraData
        
    def getMagnitude(self,cVal):
        """ Return the magnitude of complex variable cVal.
        Note actually returns magnitude ^2 for consistency with 
        pebble implementation!
        """
        #print(cVal)
        sumSq = cVal.real * cVal.real + cVal.imag * cVal.imag
        #print(cVal,"sumSq=%f, abs=%f" % (sumSq, np.abs(cVal)))
        return(sumSq)

########################################################################
# Main Program
########################################################################
if (__name__ == "__main__"):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inFile", required=True,
                    help="input data file (or folder)")
    ap.add_argument("-c", "--configFile", required = False,
                    help="configuration file name (to override defaults)")

    ap.add_argument("-p", "--plot", dest='plot', action='store_true',
                    help="Plot the data as it is processed")
    args = vars(ap.parse_args())

    print(args)

    fname = args['inFile']

    if (args['configFile'] is not None):
        config = readConfig(args['configFile'])
    else:
        config = defaultConfig

    print("config=",config)

    timeArr, alarmStateArr, hrArr, accArr, \
        fftArr, fftFreq = osdDataUtils.getOsdData(fname,
                                                  config['nSamp'])

    print("accArr", accArr.shape, accArr.dtype, accArr[0, 0].dtype)
    print("fftArr", fftArr.shape)
    print("fftFreq", fftFreq, fftFreq.shape)


    nAlarmCorrect = 0
    nAlarmWrong = 0
    
    pm = PebbleModel(config)
    for i in range(0, accArr.shape[0]):
        # print("i=%d of %d" % (i,accArr.shape[0]))
        row = accArr[i]
        # print row
        alarmState, extraData = pm.doAnalysis(row, plotData = args['plot'])


        if (alarmStateArr[i].strip() == "ALARM"):
            if (alarmState == 2):
                print("Alarm Detected Correctly")
                nAlarmCorrect += 1
            else:
                print("Failed to Detect Alarm")
                print(timeArr[i],alarmState,alarmStateArr[i],
                      "specPower=%.0f, roiPower=%.0f, ratio=%.1f"
                      % (extraData['specPower'],
                         extraData['roiPower'],
                         extraData['roiRatio']))
                if (args['plot']):
                    fig, ax = plt.subplots()
                    ax.plot(extraData['fftFreq'][0:config['nSamp']/2],
                            np.abs(extraData['fftArr'][0:config['nSamp']/2]))
                    plt.show()

                nAlarmWrong += 1
        else:
            print("%s is not an alarm state" % alarmStateArr[i])

    print("nAlarmCorrect = %d, nAlarmWrong=%d - Reliability=%f" %
          (nAlarmCorrect, nAlarmWrong,
           1.0*nAlarmCorrect/(nAlarmCorrect+nAlarmWrong)))
        # exit(-1)
