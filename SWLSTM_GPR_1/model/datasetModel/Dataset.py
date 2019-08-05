# -*- coding: utf-8 -*-
# @File         : Dataset.py
# @Author       : Zhendong Zhang
# @Email        : zzd_zzd@hust.edu.cn
# @University   : Huazhong University of Science and Technology
# @Date         : 2019/7/30
# @Software     : PyCharm
# -*---------------------------------------------------------*-

import numpy as np


class Dataset(object):

    def __init__(self, series, featureIndex, sepRatio):
        self.series = series
        self.featureIndex = featureIndex
        self.sepRatio = sepRatio
        self.seriesMin = series.min()
        self.seriesMax = series.max()
        self.normaSeries = self.normalizeWithMinMax()
        self.datasetX, self.datasetY = self.createDatasetWithFeatureIndex()
        self.dataset = np.hstack((self.datasetX, self.datasetY))
        self.trainX, self.trainY, self.validationX, self.validationY = self.sepTrainAndValidation()

    # normalizaiton
    def normalizeWithMinMax(self):
        normaSeries = (self.series - self.seriesMin)/(self.seriesMax - self.seriesMin)
        return normaSeries

    # restore
    def restoreWithMinMax(self, normaSeries):
        series = self.seriesMin + normaSeries*(self.seriesMax - self.seriesMin)
        return series

    def createDatasetWithFeatureIndex(self):
        maxIndex = self.featureIndex.max()
        featureNum = len(self.featureIndex)
        seriesNum = len(self.series)
        datasetNum = seriesNum - maxIndex

        datasetX = np.zeros(shape=(datasetNum, featureNum))
        datasetY = np.zeros(shape=(datasetNum, 1))

        normaSeries = self.normaSeries
        for index, i in zip(range(maxIndex, seriesNum), range(datasetNum)):
            datasetY[i, 0] = normaSeries[index]
            for j in range(featureNum):
                datasetX[i, j] = normaSeries[index - self.featureIndex[j]]

        return datasetX, datasetY

    # divide training set and validaiton set
    def sepTrainAndValidation(self):
        datasetNum = self.dataset.shape[0]
        sepIndex = round(datasetNum*self.sepRatio)
        trainX = self.datasetX[0:sepIndex, :]
        trainY = self.datasetY[0:sepIndex, :]
        validationX = self.datasetX[sepIndex:datasetNum, :]
        validationY = self.datasetY[sepIndex:datasetNum, :]
        return trainX, trainY, validationX, validationY




