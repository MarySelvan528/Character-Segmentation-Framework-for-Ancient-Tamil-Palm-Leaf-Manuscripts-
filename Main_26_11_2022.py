import os
from random import uniform

import cv2 as cv
import numpy as np
from numpy import matlib

from AVOA import AVOA
from CSO import CSO
from Global_Vars import Global_Vars
from HHO import HHO
from Image_Results import Image_Results
from LBP_Pattern import LBP
from LTRP_Pattern import LTRP_Pattern
from Model_AE import Model_AutoEncoder
from Models import Model_Ensemble, Model_ANN, Model_SVM, Model_BL
from Objfun import objfun
from Plot_Results import *
from Proposed import Proposed


def Read_Image(filename):
    image = cv.imread(filename)
    image = np.uint8(image)
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    # image = cv.resize(image, (512, 512))
    return image


def Read_Dataset(Directory):
    Images = []
    listImages = os.listdir(Directory)
    for i in range(len(listImages)):
        filename = Directory + listImages[i]
        image = Read_Image(filename)
        Images.append(image)
    return Images


# Read Dataset
an = 0
if an == 1:
    Images = Read_Dataset('./Images/')
    np.save('Images.npy', Images)

# Preprocessing
an = 0
if an == 1:
    Images = np.load('Images.npy', allow_pickle=True)
    Preprocess = []
    for i in range(Images.shape[0]):
        print(i)
        image = Images[i]
        alpha = 1.5  # Contrast control
        beta = 10  # Brightness control
        adjusted = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
        Thresh = cv.threshold(adjusted, 75, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        Preprocess.append(Thresh)
        # Prep = np.zeros(image.shape, dtype=np.uint8)
        # Thresh = cv.threshold(image, 75, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        # kernel = np.ones((3, 3), np.uint8)
        # # opening = cv.morphologyEx(Thresh, cv.MORPH_OPEN, kernel, iterations=1)
        # # closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=1)
        # # Prep[closing == 255] = image[closing == 255]
        # Prep[Thresh == 255] = image[Thresh == 255]
        # filtered = cv.medianBlur(Prep, 7)
        # Preprocess.append(filtered)
    np.save('Preprocess.npy', np.array(Preprocess, dtype=object))

# Line Segmentation
an = 0
if an == 1:
    Target = []
    Segment = []
    Images = np.load('Images.npy', allow_pickle=True)
    for i in range(Images.shape[0]):
        image = Images[i]
        threshold = cv.threshold(image, 75, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        analysis = cv.connectedComponentsWithStats(threshold, 4, cv.CV_32S)
        (totalLabels, label_ids, values, centroid) = analysis
        output = np.zeros(image.shape, dtype="uint8")
        for j in range(1, totalLabels):
            print(i, j, totalLabels)
            area = values[j, cv.CC_STAT_AREA]
            componentMask = (label_ids == j).astype("uint8") * 255
            output = cv.bitwise_or(output, componentMask)
            index = np.where(componentMask != 0)
            x = np.min(index[0])
            y = np.min(index[1])
            w = np.max(index[0]) - x
            h = np.max(index[1]) - y
            image = np.zeros((w + 11, h + 11))
            image[index[0] - x + 5, index[1] - y + 5] = 255
            Segment.append(image)
            if 25 < area <= 1000:
                Target.append(1)
            else:
                Target.append(0)
    np.save('Segment.npy', Segment)
    np.save('Target.npy', Target)

# Feature Extraction
an = 0
if an == 1:
    Segment = np.load('Segment.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    for i in range(Segment.shape[0]):
        seg = Segment[i]
        ltrp = LTRP_Pattern(seg)
        lbp = LBP(seg)
    pred, Feature = Model_AutoEncoder(Segment, Target)
    np.save('Feature.npy', Feature)

an = 0
if an == 1:
    Feature = np.load('Feature.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)

    Global_Vars.Data = Feature
    Global_Vars.Target = Target

    Npop = 10
    Chlen = 4
    xmin = matlib.repmat(np.concatenate([2.0, 30, 30, 1e-4], axis=None), Npop, 1)
    xmax = matlib.repmat(np.concatenate([20.0, 300, 300, 1e-1], axis=None), Npop, 1)
    initsol = np.zeros(xmax.shape)
    for p1 in range(Npop):
        for p2 in range(xmax.shape[1]):
            initsol[p1, p2] = uniform(xmin[p1, p2], xmax[p1, p2])
    fname = objfun
    Max_iter = 25

    print("HHO...")
    [bestfit1, fitness1, bestsol1, time1] = HHO(initsol, fname, xmin, xmax, Max_iter)

    print("AVOA...")
    [bestfit2, fitness2, bestsol2, time2] = AVOA(initsol, fname, xmin, xmax, Max_iter)

    print("SOA...")
    [bestfit3, fitness3, bestsol3, time3] = CSO(initsol, fname, xmin, xmax, Max_iter)

    print("FA...")
    [bestfit4, fitness4, bestsol4, time4] = CSO(initsol, fname, xmin, xmax, Max_iter)

    print("Proposed")
    [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)
    Bestsol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    np.save('Bestsol.npy', Bestsol)

# Classification by varieying Learning percentage
an = 0
if an == 1:
    Bestsol = np.load('Bestsol.npy', allow_pickle=True)
    Learnper = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    Target = np.load('Target.npy', allow_pickle=True)
    Features = np.load('Feature.npy', allow_pickle=True)
    EVAL = []
    for i in range(len(Learnper)):
        Eval = np.zeros((10, 14))
        for j in range(Bestsol.shape[0]):
            sol = Bestsol[j, :]
            learnperc = round(Features.shape[0] * Learnper[i])
            Train_Data = Features[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Features[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            Eval[j, :] = Model_Ensemble(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        Eval[5, :] = Model_ANN(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[6, :] = Model_SVM(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[7, :] = Model_BL(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[8, :] = Model_Ensemble(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[9, :] = Model_Ensemble(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        EVAL.append(Eval)
    np.save('Eval_all.npy', EVAL)


plot_results()
plotConvResults()
Image_Results()
