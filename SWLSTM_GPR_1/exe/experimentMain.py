# -*- coding: utf-8 -*-
# @File         : fdsfd.py
# @Author       : Zhendong Zhang
# @Email        : zzd_zzd@hust.edu.cn
# @University   : Huazhong University of Science and Technology
# @Date         : 2019/7/29
# @Software     : PyCharm
# -*---------------------------------------------------------*-

import numpy as np
import pandas as pd
from scipy import stats
from SWLSTM_GPR_1.model.implModel.ModelFactory import LSTMmodel, SWLSTMmodel, GRUmodel, GPRmodel
from SWLSTM_GPR_1.evaluation.Evaluation import EvaluationUtils
from SWLSTM_GPR_1.evaluation.Draw import DrawUtils
from SWLSTM_GPR_1.model.datasetModel.Dataset import Dataset
from datetime import datetime
import time
import os


if __name__ == "__main__":
    # base parameter settings
    basePath = '../data/sampleWindspeed.xlsx'
    data = pd.read_excel(basePath, header=None)

    resultSaveBasePath = '../results/'
    plotSaveBasePath = '../plots/'
    timeFlag = int(time.time())

    experimentConfigSampleIndex = [0, 1, 2, 3]
    experimentConfigModelNames = ['SWLSTM-GPR', 'SWLSTM', 'LSTM', 'GRU', 'GPR']
    experimentConfigPointMetricNames = ['RMSE', 'R2']
    experimentConfigIntervalMetricNames = ['CP', 'MWP', 'MC']
    experimentConfigProbaMetricNames = ['CRPS']
    experimentConfigReliabMetricNames = ['PIT']

    isSaveFigure = True
    isShowFigure = True
    isCalMetricByNormalization = [0, 0, 1]

    datasetNum = len(experimentConfigSampleIndex)
    modelNum = len(experimentConfigModelNames)
    pointMetricNum = len(experimentConfigPointMetricNames)+1
    intervalMetricNum = len(experimentConfigIntervalMetricNames)
    probaMetricNum = len(experimentConfigProbaMetricNames)

    experimentPointMetrics = np.zeros(shape=(modelNum, datasetNum * pointMetricNum))
    experimentIntervalMetrics = np.zeros(shape=(modelNum, datasetNum * pointMetricNum))
    experimentProbaMetrics = np.zeros(shape=(modelNum, datasetNum * probaMetricNum))

    # predictions
    for sampleIndex in experimentConfigSampleIndex:
        series = data[sampleIndex]
        series = series.dropna()
        series = np.array(series)

        if sampleIndex == 0:
            featureIndex = np.array([9, 8, 6, 5, 4, 3, 2, 1])  # 1
        elif sampleIndex == 1:
            featureIndex = np.array([9, 7, 6, 4, 3, 2, 1])  # 2
        elif sampleIndex == 2:
            featureIndex = np.array([6, 5, 4, 3, 2, 1])  # 3
        else:
            featureIndex = np.array([9, 6, 5, 4, 3, 2, 1])  # 4
        dataset = Dataset(series, featureIndex, 0.8)

        parameter = dict()

        parameter['nodeNums'] = [8]
        parameter['lossName'] = 'mse'
        parameter['optimizer'] = 'adam'
        parameter['batchSize'] = 32
        parameter['epochs'] = 2000
        parameter['metrics'] = []  # [metrics.mape, metrics.msle]

        resultSavePath = resultSaveBasePath + str(timeFlag) + '/'
        if not os.path.exists(resultSavePath):
            os.makedirs(resultSavePath)
        resultSavePath = resultSavePath + 'dataset ' + str(sampleIndex + 1) + '_results_' + str(timeFlag) + '.xlsx'
        writer = pd.ExcelWriter(resultSavePath)

        modelIndex = 0
        for modelName in experimentConfigModelNames:
            trainX = dataset.trainX
            trainY = dataset.trainY
            validationX = dataset.validationX
            validationY = dataset.validationY
            if modelName == 'SWLSTM-GPR' or modelName == 'LSTM' or modelName == 'GRU':
                trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
                trainY = trainY.reshape(trainY.shape[0], 1, 1)
                validationX = validationX.reshape(validationX.shape[0], 1, validationX.shape[1])
                validationY = validationY.reshape(validationY.shape[0], 1, 1)

            if modelName == 'SWLSTM-GPR':
                model = SWLSTMmodel(trainX, trainY, validationX, validationY, parameter)
            elif modelName == 'LSTM':
                model = LSTMmodel(trainX, trainY, validationX, validationY, parameter)
            elif modelName == 'GRU':
                model = GRUmodel(trainX, trainY, validationX, validationY, parameter)
            elif modelName == 'GPR':
                model = GPRmodel(trainX, trainY, validationX, validationY, parameter)

            model.constructModel()
            startTime = datetime.now()
            model.fit()
            endTime = datetime.now()
            tt = (endTime - startTime).seconds
            predictions, sigmas = model.predict()

            if modelName == 'SWLSTM-GPR':
                trainX2, notImportant = model.predict(trainX)
                trainY2 = trainY
                validationX2, notImportant = model.predict(validationX)
                validationY2 = validationY

                trainX2 = dataset.restoreWithMinMax(trainX2[:, 0, :])
                trainY2 = dataset.restoreWithMinMax(trainY2[:, 0, :])
                validationX2 = dataset.restoreWithMinMax(validationX2[:, 0, :])
                validationY2 = dataset.restoreWithMinMax(validationY2[:, 0, :])

                model2 = GPRmodel(trainX2, trainY2, validationX2, validationY2)

                model2.constructModel()
                model2.fit()
                predictions2, sigmas2 = model2.predict()

                predictions = predictions2
                sigmas = sigmas2

                predictions = (predictions - dataset.seriesMin) / (dataset.seriesMax - dataset.seriesMin)
                sigmas = sigmas / (dataset.seriesMax - dataset.seriesMin)

            predictions = predictions.flatten()
            y = validationY.flatten()

            predictionsNormalization = predictions
            yNormalization = y
            predictions = dataset.restoreWithMinMax(predictionsNormalization)
            y = dataset.restoreWithMinMax(yNormalization)

            # calculate metrics
            eva = EvaluationUtils()
            if isCalMetricByNormalization[0]:
                pointMetrics = eva.getPointPredictionMetric(predictionsNormalization, yNormalization,
                                                            metricNames=experimentConfigPointMetricNames, isPrint=True)
            else:
                pointMetrics = eva.getPointPredictionMetric(predictions, y, metricNames=experimentConfigPointMetricNames,
                                                            isPrint=True)
            print('TT ：' + str(tt) + 's')
            pointMetricIndex = 0
            pointMetricStartIndex = sampleIndex * pointMetricNum
            for pointMetricName in experimentConfigPointMetricNames:
                experimentPointMetrics[modelIndex, pointMetricStartIndex+pointMetricIndex] = pointMetrics[pointMetricName]
                pointMetricIndex = pointMetricIndex+1
            experimentPointMetrics[modelIndex, pointMetricStartIndex + pointMetricIndex] = tt

            plotSavePath = plotSaveBasePath + str(timeFlag) + '/'
            if not os.path.exists(plotSavePath):
                os.makedirs(plotSavePath)
            datasetAndModelFlag = 'dataset ' + str(sampleIndex + 1) + '_' + modelName + '_' + str(timeFlag)
            plotResultSavePath = plotSavePath + datasetAndModelFlag + '_predictions.jpg'
            plotPitSavePath = plotSavePath + datasetAndModelFlag + '_pit.jpg'

            locArray = np.array([[1.0, 0.35], [1.0, 0.35], [0.3, 0.35], [0.3, 0.3]])
            if modelName == 'SWLSTM-GPR' or modelName == 'GPR':
                sigmasNormalization = sigmas
                sigmas = sigmasNormalization * (dataset.seriesMax - dataset.seriesMin)
                lowerNormalization = predictionsNormalization - 1.96 * sigmasNormalization
                upperNormalization = predictionsNormalization + 1.96 * sigmasNormalization
                lower = predictions - 1.96 * sigmas
                upper = predictions + 1.96 * sigmas
                if isCalMetricByNormalization[1]:
                    intervalMetrics = eva.getIntervalPredictionMetric(lowerNormalization, upperNormalization,
                                                                      yNormalization,
                                                                      metricNames=experimentConfigIntervalMetricNames,
                                                                      isPrint=True)
                else:
                    intervalMetrics = eva.getIntervalPredictionMetric(lower, upper, y,
                                                                      metricNames=experimentConfigIntervalMetricNames,
                                                                      isPrint=True)

                if isCalMetricByNormalization[2]:
                    probaMetrics = eva.getProbabilityPredictionMetric(predictionsNormalization, sigmasNormalization,
                                                                      yNormalization,
                                                                      metricNames=experimentConfigProbaMetricNames,
                                                                      isPrint=True)
                else:
                    probaMetrics = eva.getProbabilityPredictionMetric(predictions, sigmas, y,
                                                                      metricNames=experimentConfigProbaMetricNames,
                                                                      isPrint=True)
                reliabilityMetrics = eva.getReliabilityMetric(predictions, sigmas, y,
                                                              metricNames=experimentConfigReliabMetricNames)

                intervalMetricIndex = 0
                intervalMetricStartIndex = sampleIndex * intervalMetricNum
                for intervalMetricName in experimentConfigIntervalMetricNames:
                    experimentIntervalMetrics[modelIndex, intervalMetricStartIndex + intervalMetricIndex] = intervalMetrics[intervalMetricName]
                    intervalMetricIndex = intervalMetricIndex + 1

                probaMetricIndex = 0
                probaMetricStartIndex = sampleIndex * probaMetricNum
                for probaMetricName in experimentConfigProbaMetricNames:
                    experimentProbaMetrics[modelIndex, probaMetricStartIndex + probaMetricIndex] = probaMetrics[probaMetricName]
                    probaMetricIndex = probaMetricIndex + 1

                # plot
                drawUtils = DrawUtils()
                drawUtils.drawPredictions(predictions, y, lower, upper, alpha='90%', isInterval=True, xlabel='period',
                                          ylabel='wind speed(m/s)', title='dataset '+str(sampleIndex+1),
                                          legendLoc=locArray[sampleIndex, :], isSave=isSaveFigure,
                                          savePath=plotResultSavePath, isShow=isShowFigure)
                drawUtils.drawPIT(reliabilityMetrics[experimentConfigReliabMetricNames[0]], cdf=stats.uniform,
                                  xlabel='uniform distribution', ylabel='PIT', title='dataset '+str(sampleIndex+1),
                                  isSave=isSaveFigure, savePath=plotPitSavePath, isShow=isShowFigure)
            else:
                drawUtils.drawPredictions(predictions, y, None, None, alpha='--', isInterval=False, xlabel='period',
                                          ylabel='wind speed(m/s)', title='dataset '+str(sampleIndex+1),
                                          legendLoc=locArray[sampleIndex, :], isSave=isSaveFigure,
                                          savePath=plotResultSavePath, isShow=isShowFigure)
                sigmas = np.zeros(shape=predictions.shape)

            resultsDataFrame = pd.DataFrame(np.hstack((predictions.reshape(len(predictions), 1), y.reshape(len(y), 1),
                                                       sigmas.reshape(len(sigmas), 1))),
                                            columns=['predictions', 'observations', 'sigmas'])
            resultsDataFrame.to_excel(writer, modelName)

            modelIndex = modelIndex+1
        writer.save()
        writer.close()
    # save results
    experimentConfigPointMetricNamesAll = experimentConfigPointMetricNames
    experimentConfigPointMetricNamesAll.append('TT')
    experimentPointMetricsDataFrame = pd.DataFrame(experimentPointMetrics,
                                                   columns=np.tile(experimentConfigPointMetricNamesAll, datasetNum),
                                                   index=experimentConfigModelNames)
    experimentIntervalMetricsDataFrame = pd.DataFrame(experimentIntervalMetrics,
                                                      columns=np.tile(experimentConfigIntervalMetricNames, datasetNum),
                                                      index=experimentConfigModelNames)
    experimentProbaMetricsDataFrame = pd.DataFrame(experimentProbaMetrics,
                                                   columns=np.tile(experimentConfigProbaMetricNames, datasetNum),
                                                   index=experimentConfigModelNames)
    metricSavePath = resultSaveBasePath + str(timeFlag) + '/'
    if not os.path.exists(metricSavePath):
        os.makedirs(metricSavePath)
    metricSavePath = metricSavePath + 'metrics_' + str(timeFlag) + '.xlsx'
    writer = pd.ExcelWriter(metricSavePath)
    experimentPointMetricsDataFrame.to_excel(writer, 'pointMetrics')
    experimentIntervalMetricsDataFrame.to_excel(writer, 'intervalMetrics')
    experimentProbaMetricsDataFrame.to_excel(writer, 'probaMetrics')
    writer.save()
    writer.close()

