# -*- coding: utf-8 -*-
# @File         : __init__.py.py
# @Author       : Zhendong Zhang
# @Email        : zzd_zzd@hust.edu.cn
# @University   : Huazhong University of Science and Technology
# @Date         : 2019/7/29
# @Software     : PyCharm
# -*---------------------------------------------------------*-


import numpy as np
from scipy.stats import norm


class EvaluationUtils(object):

    # obtain point prediction metrics
    def getPointPredictionMetric(self, predictions, observations, metricNames=None, isPrint=False):
        if metricNames is None:
            metricNames = ['MAE', 'MSE', 'RMSE', 'MAPE', 'R2']
        metrics = {}
        for metricName in metricNames:
            metric = None
            if metricName == 'MAE':
                metric = EvaluationUtils.getMAE(predictions, observations)
            elif metricName == 'MSE':
                metric = EvaluationUtils.getMSE(predictions, observations)
            elif metricName == 'RMSE':
                metric = EvaluationUtils.getRMSE(predictions, observations)
            elif metricName == 'MAPE':
                metric = EvaluationUtils.getMAPE(predictions, observations)
            elif metricName == 'R2':
                metric = EvaluationUtils.getRsquare(predictions, observations)
            else:
                raise Exception('unknown point prediction metric name: '+metricName)

            if isPrint:
                print(metricName + ' : ' + str(metric))
            metrics[metricName] = metric
        return metrics

    @staticmethod
    def getMAE(predictions, observations):
        MAE = np.mean(np.abs(predictions-observations))
        return MAE

    @staticmethod
    def getMSE(predictions, observations):
        MSE = np.mean(np.power(predictions-observations, 2))
        return MSE

    @staticmethod
    def getRMSE(predictions, observations):
        MSE = EvaluationUtils.getMSE(predictions, observations)
        RMSE = np.sqrt(MSE)
        return RMSE

    @staticmethod
    def getMAPE(predictions, observations):
        MAPE = np.mean(np.true_divide(np.abs(predictions-observations), np.abs(observations)))
        return MAPE

    @staticmethod
    def getRsquare(predictions, observations):
        mean = np.mean(observations)
        numerator = np.sum(np.power(observations-predictions, 2))
        denominator = np.sum(np.power(observations-mean, 2))
        Rsquare = 1-numerator/denominator
        return Rsquare

    # obtain interval prediction metrics
    def getIntervalPredictionMetric(self, lower, upper, observations, metricNames=None, isPrint=False):
        if metricNames==None:
            metricNames = ['CP', 'MWP', 'CM']
        metrics = {}
        for metricName in metricNames:
            metric = None
            if metricName == 'CP':
                metric = EvaluationUtils.getCP(lower, upper, observations)
            elif metricName == 'MWP':
                metric = EvaluationUtils.getMWP(lower, upper, observations)
            elif metricName == 'CM':
                metric = EvaluationUtils.getCM(lower, upper, observations)
            elif metricName == 'MC':
                metric = 1.0/EvaluationUtils.getCM(lower, upper, observations)
            else:
                raise Exception('unknown interval prediction metric name: '+metricName)
            metrics[metricName] = metric
            if isPrint:
                print(metricName + ' : ' + str(metric))
        return metrics

    @staticmethod
    def getCP(lower, upper, observations):
        N = observations.shape[0]
        count = 0
        for i in range(N):
            if observations[i]>=lower[i] and observations[i]<=upper[i]:
                count = count + 1
        CP = count/N
        return CP

    @staticmethod
    def getMWP(lower, upper, observations):
        N = observations.shape[0]
        MWP = 0
        for i in range(N):
            if upper[i]<lower[i]:
                print(i)
            MWP = MWP + (upper[i]-lower[i])/np.abs(observations[i])
        MWP = MWP/N
        return MWP

    @staticmethod
    def getCM(lower, upper, observations):
        CM = EvaluationUtils.getCP(lower, upper, observations)/EvaluationUtils.getMWP(lower, upper, observations)
        return CM

    # obtain probability prediction metrics
    def getProbabilityPredictionMetric(self, predictions, sigmas, observations, metricNames=None, isPrint=False):
        if metricNames is None:
            metricNames = ['CRPS']
        metrics = {}
        for metricName in metricNames:
            metric = None
            if metricName == 'CRPS':
                metric = EvaluationUtils.getCRPS(predictions, sigmas, observations)
            else:
                raise Exception('unknown probability prediction metric name: '+metric)
            metrics[metricName] = metric
            if isPrint:
                print(metricName + ' : ' + str(metric))
        return metrics

    @staticmethod
    def getCRPS(predictions, sigmas, observations):
        # only for gaussian distribution
        num = len(observations)
        areas = np.zeros(shape=(num, 1))
        for i in range(num):
            x1 = norm.ppf(0.9999, loc=predictions[i], scale=sigmas[i])
            x0 = norm.ppf(0.0001, loc=predictions[i], scale=sigmas[i])

            if x1 < observations[i]:
                x1 = observations[i]

            if x0 > observations[i]:
                x0 = observations[i]

            x = np.linspace(x0, x1, 1000)
            y = np.zeros(shape=x.shape)
            area = 0.0
            for j in range(len(x)):
                y[j] = norm.cdf(x[j], loc=predictions[i], scale=sigmas[i]) - EvaluationUtils.getH(x[j], observations[i])
                y[j] = np.power(y[j], 2)
                if j >= 1:
                    area = area + (y[j]+y[j-1])*(x[j]-x[j-1])/2
            areas[i, 0] = area
        crps = np.mean(areas)
        return crps

    @staticmethod
    def getH(prediction, observation):
        if prediction<observation:
            return 0
        else:
            return 1

    # obtain reliability metrics
    def getReliabilityMetric(self, predictions, sigmas, observations, metricNames=None):
        if metricNames is None:
            metricNames = ['PIT']
        metrics = {}
        for metricName in metricNames:
            metric = None
            if metricName == 'PIT':
                metric = EvaluationUtils.getPIT(predictions, sigmas, observations,)
            else:
                raise Exception('unknown probability prediction metric name: '+metric)
            metrics[metricName] = metric
        return metrics

    @staticmethod
    def getPIT(predictions, sigmas, observations,):
        PIT = np.zeros(shape=observations.shape)
        for i in range(observations.shape[0]):
            PIT[i] = norm.cdf(observations[i], loc=predictions[i], scale=sigmas[i])
        return PIT

