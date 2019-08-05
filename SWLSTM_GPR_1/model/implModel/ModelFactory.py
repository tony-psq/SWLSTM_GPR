# -*- coding: utf-8 -*-
# @File         : ModelFactory.py
# @Author       : Zhendong Zhang
# @Email        : zzd_zzd@hust.edu.cn
# @University   : Huazhong University of Science and Technology
# @Date         : 2019/7/29
# @Software     : PyCharm
################################################################

import abc
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from SWLSTM_GPR_1.model.recurrent import LSTM, SWLSTM, GRU
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic as RQ


class RecurrentModel(object):

    # constructor
    def __init__(self, trainX, trainY, validationX, validationY, hyperParameters=None):
        self.trainX = trainX
        self.trainY = trainY
        self.validationX = validationX
        self.validationY = validationY
        self.hyperParameters = hyperParameters
        self.recurrentModel = None

    # construct model based on hyperParameters
    @abc.abstractmethod
    def constructModel(self):
        pass

    # train model
    def fit(self):
        batchSize = self.hyperParameters['batchSize']
        epochs = self.hyperParameters['epochs']
        optimizer = self.hyperParameters['optimizer']
        metrics = self.hyperParameters['metrics']
        lossName = self.hyperParameters['lossName']

        adam = Adam(lr=0.01)

        self.recurrentModel.compile(loss=lossName, optimizer=adam, metrics=metrics)
        reduceLr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=50,
                                     verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.0001)

        self.recurrentModel.fit(self.trainX, self.trainY, batch_size=batchSize, epochs=epochs,
                                )# callbacks=[reduceLr]

    # predict for validationX
    def predict(self, validationX=None):
        if validationX is None:
            validationX = self.validationX
        predictions = self.recurrentModel.predict(validationX)
        return predictions, None


class LSTMmodel(RecurrentModel):

    # construct model based on hyperParameters
    def constructModel(self):
        nodeNums = self.hyperParameters['nodeNums']
        recurrentModel = Sequential()
        recurrentModel.add(LSTM(units=nodeNums[0], return_sequences=True))
        recurrentModel.add(Dense(1, activation='sigmoid'))
        self.recurrentModel = recurrentModel

class SWLSTMmodel(RecurrentModel):
    # construct model based on hyperParameters
    def constructModel(self):
        nodeNums = self.hyperParameters['nodeNums']
        optimizer = self.hyperParameters['optimizer']
        metrics = self.hyperParameters['metrics']
        lossName = self.hyperParameters['lossName']

        recurrentModel = Sequential()
        recurrentModel.add(SWLSTM(units=nodeNums[0], return_sequences=True))
        recurrentModel.add(Dense(1, activation='sigmoid'))
        recurrentModel.compile(loss=lossName, optimizer=optimizer, metrics=metrics)
        self.recurrentModel = recurrentModel


class GRUmodel(RecurrentModel):
    # construct model based on hyperParameters
    def constructModel(self):
        nodeNums = self.hyperParameters['nodeNums']
        optimizer = self.hyperParameters['optimizer']
        metrics = self.hyperParameters['metrics']
        lossName = self.hyperParameters['lossName']

        recurrentModel = Sequential()
        recurrentModel.add(GRU(units=nodeNums[0], return_sequences=True))
        recurrentModel.add(Dense(1, activation='sigmoid'))
        recurrentModel.compile(loss=lossName, optimizer=optimizer, metrics=metrics)
        self.recurrentModel = recurrentModel


class GPRmodel(RecurrentModel):
    def constructModel(self):
        kernel = RQ(1.0, 1.0, (1e-5, 1e5), (1e-5, 1e5))
        self.recurrentModel = GaussianProcessRegressor(kernel=kernel)

    def fit(self):
        self.recurrentModel.fit(self.trainX, self.trainY)

    # predict for validationX
    def predict(self, validationX=None):
        if validationX is None:
            validationX = self.validationX
        predictions, sigmas = self.recurrentModel.predict(validationX, return_std=True)
        return predictions, sigmas


