import json
import random
import struct

import commonFunctions
import constructNetWork_MultiChannelGCNN as TC
import GraphData_IO
import cPickle as p

import write_param
from Graph import Graph
from database import CodeChef
from nn import serialize

import sys, os
import gcnn_params as params

sys.path.append('../nn')
datapath = params.datapath

from InitParam import *

import numpy as np

sys.setrecursionlimit(1000000)

word_dict, vectors, numFea = GraphData_IO.LoadVocab(vocabfile=datapath + 'tokvec.txt')
print 'Load token embedding from: ', datapath + 'tokvec.txt'
print 'Vocab length =', len(word_dict)
print 'Embedding size =', numFea

numDis = params.numDis
numOut = params.numOut

numMap = params.numMap
numCon = params.numCon
numView = params.numView

np.random.seed(314)

preBword = vectors # numFea*len(Vocab) # word vectors
print '(Words = ', len(word_dict), ') * (Size =', numFea,') = ', len(preBword)
# Initialize weights and biases
Weights = np.array([])
Biases = np.array([])

# word/symbol/token embedding
Biases, BwordIdx = InitParam(Biases, newWeights=preBword)
#parameters include
# word_dict: [dict_view1, dict_view2]
# numView, numFea, numCon, numDis, numOut, \
# Wconv_root    [View1:[conv1, conv2, ...], View2:[conv1, conv2, ...]]
# Wconv_in  [View1:[conv1, conv2, ...], View2:[conv1, conv2, ...]]
# Wconv_out     [View1:[conv1, conv2, ...], View2:[conv1, conv2, ...]]
# Bconv         [View1:[conv1, conv2, ...], View2:[conv1, conv2, ...]]
# Wdis[pool_view1, pool_view2, ...---> Dis]

Wmap =[]
Bmap =[]

Wconv_root=[]
Wconv_in =[]
Wconv_out =[]
Bconv =[]

# mapping layers
tmpnumMap =[numFea]
tmpnumMap.extend(numMap)
for m in range(1, len(numMap)+1):
    wmap_view =[]
    bmap_view =[]
    for v in range(0, numView):
        Weights, w = InitParam(Weights, num=tmpnumMap[m-1] * tmpnumMap[m])
        wmap_view.append(w)

        Biases, b = InitParam(Biases, num=tmpnumMap[m])
        bmap_view.append(b)

    Wmap.append(wmap_view)
    Bmap.append(bmap_view)

# convolution layers
num_Pre = tmpnumMap[-1]

for c in range(len(numCon)):
    if c==0:
        view_wroot =[None]*numView
        view_win = [None]*numView
        view_wout = [None] * numView

        for v in range(0, numView):
            Weights, w = InitParam(Weights, num=num_Pre * numCon[c])
            view_wroot[v]= w

            Weights, w = InitParam(Weights, num=num_Pre * numCon[c])
            view_win[v] = w

            Weights, w = InitParam(Weights, num=num_Pre * numCon[c])
            view_wout[v] = w

        Wconv_root.append(view_wroot)
        Wconv_in.append(view_win)
        Wconv_out.append(view_wout)
    else:
        Weights, w = InitParam(Weights, num=num_Pre * numCon[c])
        Wconv_root.append(w)

        Weights, w = InitParam(Weights, num=num_Pre * numCon[c])
        Wconv_in.append(w)

        Weights, w = InitParam(Weights, num=num_Pre * numCon[c])
        Wconv_out.append(w)

    Biases, b = InitParam(Biases, num=numCon[c])
    Bconv.append(b)

    num_Pre = numCon[c]

# discriminative layer
Weights, Wdis = InitParam(Weights, num=num_Pre * numDis)

Biases, Bdis = InitParam(Biases, num=numDis)

# output layer
Weights, Woutput = InitParam(Weights, num=numDis * numOut, upper=.0002, lower=-.0002)
Biases, Boutput = InitParam(Biases, newWeights=np.zeros((numOut, 1)))

Weights = Weights.reshape((-1, 1))
Biases = Biases.reshape((-1, 1))

print 'num of biases = ', len(Biases)
print 'num of weights = ', len(Weights)

# initial the gradients

gradWeights = np.zeros_like(Weights)
gradBiases = np.zeros_like(Biases)

print 'numCon', numCon
print 'numDis', numDis
print 'numOut', numOut
print 'Weights', len(Weights)
print 'Bias', len(Biases)
# dwadwad
# 17940
# 1544
#
paramFile ='paramTest_ASTGraph_V'+str(numView)
write_param.write_binary(datapath+paramFile, Weights, Biases)
print 'Parameters have been saved at: ', datapath+paramFile
problem =''
def InitByNodes(graph, word_dict):
    # Embedding ---> Mapping ---> Conv1 ---> .... ---> Convn ---> Pooling ---> Fully-Connected ---> Output
    # ConstructGraphConvolution(graph, word_dict, numView, numFea, numMap, numCon, numDis, numOut, \
    #                               Wmap, Bmap,
    #                               Wconv_root, Wconv_in, Wconv_out, Bconv, \
    #                               Wdis, Woutput, Bdis, Boutput
    #                               ):

    # word_dict: [dict_view1, dict_view2]
    # numView, numFea, numCon, numDis, numOut, \
    # Wconv_root    [View1:[conv1, conv2, ...], View2:[conv1, conv2, ...]]
    # Wconv_income  [View1:[conv1, conv2, ...], View2:[conv1, conv2, ...]]
    # Wconv_out     [View1:[conv1, conv2, ...], View2:[conv1, conv2, ...]]
    # Bconv         [View1:[conv1, conv2, ...], View2:[conv1, conv2, ...]]
    # Wdis[pool_view1, pool_view2, ...---> Dis]
    layers = TC.ConstructGraphConvolution(graph, word_dict, numView, numFea, numMap, numCon, numDis, numOut, \
                                  Wmap, Bmap,
                                  Wconv_root, Wconv_in, Wconv_out, Bconv, \
                                  Wdis, Woutput, Bdis, Boutput
                                 )


    return layers

def getLabel(label):
    if label >0:
        if problem =='MNMX' or problem =='FLOW016' or problem=='SUBINC'\
                :
            return 1

    return label
def constructNetFromJson(jsonFile='', f_x ='', f_y=''):
    # open files to write
    # f_x = file(xfile, 'wb')
    # f_y = file(yfile, 'w')
    count =0
    #numinst = random.randint(20,30)
    with open(jsonFile, 'r') as f:
        jsonObjs = json.load(f)
        for obj in jsonObjs:
            count +=1
            # if count< 26086:
            #     continue
            graph = Graph.load(obj)
            g_net = InitByNodes(graph=graph, word_dict=word_dict)
            # write net
            commonFunctions.WriteNet(f_x, g_net)
            # print y
            f_y.write(str(getLabel(graph.label)) + '\n')
            if count %10:
                f_x.flush()
                f_y.flush()

    # close files
    # f_x.close()
    # f_y.close()

    return count

def CodeChefExperiment(problem = 'MNMX'):
    # tokenvec.txt and toktypeDict.txt must be put in the data directory
    datafiles = {}
    xypath = datapath + '/xy/'

    # problem: MNMX, FLOW016, SUBINC
    dtb = CodeChef(problem=problem)
    jsontrain = dtb.getTrainName()
    jsonCV = dtb.getCVName()
    jsontest = dtb.getTestName()

    print 'Construct net from:', jsontrain, jsonCV, jsontest
    datafiles['train'] = [datapath + jsontrain, xypath + dtb.problem + '_train_Xnet',
                          xypath + dtb.problem + '_train_Y.txt']
    datafiles['CV'] = [datapath + jsonCV, xypath + dtb.problem + '_CV_Xnet', xypath + dtb.problem + '_CV_Y.txt']
    datafiles['test'] = [datapath + jsontest, xypath + dtb.problem + '_test_Xnet', xypath + dtb.problem + '_test_Y.txt']

    # jsonfile, xfile, yfile = datafiles['train']
    # constructNetFromJson(jsonFile=jsonfile, xfile=xfile, yfile=yfile)
    numInst = {}
    for fold in datafiles:
        jsonfile, xfile, yfile = datafiles[fold]
        # write X, Y to file
        f_x = file(xfile, 'wb')
        f_y = file(yfile, 'w')
        numInst[fold] = constructNetFromJson(jsonFile=jsonfile, f_x=f_x, f_y=f_y)
        f_x.close()
        f_y.close()

    # write setting content
    print 'setting parameters:'
    commonFunctions.generateSettingContent(
        {'numtrain': numInst['train'], 'numcv': numInst['CV'], 'numtest': numInst['test'], 'output': numOut,
         'paramFile': paramFile, 'xtrain': dtb.problem + '_train_Xnet', 'xcv': dtb.problem + '_CV_Xnet',
         'xtest': dtb.problem + '_test_Xnet',
         'ytrain': dtb.problem + '_train_Y.txt', 'ycv': dtb.problem + '_CV_Y.txt',
         'ytest': dtb.problem + '_test_Y.txt'})

def GraphJsonKFold(K = 5, path=''):
    for idx in range(1, K+1):
        jsonObjs = []
        #get graphs from Nonvirus Dir
        gjson = GraphData_IO.getGraphsFromDataDir(path + 'Fold' + str(idx) + '/NonVirus/', classlabel=0)
        jsonObjs.extend(gjson)
        # get graphs from Virus Dir
        gjson = GraphData_IO.getGraphsFromDataDir(path+'Fold'+str(idx)+'/Virus/', classlabel=1)
        jsonObjs.extend(gjson)

        # shuffle the data
        # np.random.seed(314159)
        np.random.shuffle(jsonObjs)
        # Write to file
        with open(path+'Fold'+str(idx)+'/dataFold'+str(idx),'w') as f:
            json.dump(jsonObjs,f)
def saveXYForKFold(K=5, path=''):
    for idx in range(1, K+1):
        # write X, Y test from current folds
        f_x = open(path+ 'Fold'+str(idx)+'X_test','wb')
        f_y = open(path+ 'Fold'+str(idx)+'Y_test.txt','w')
        constructNetFromJson(path+'Fold'+str(idx)+'/dataFold'+str(idx), f_x=f_x, f_y = f_y)  # not reassign label
        f_x.close()
        f_y.close()
        # Merge other folds, Write X, Y for training
        f_x = open(path+ 'Fold'+str(idx)+'X_train','wb')
        f_y = open(path+ 'Fold'+str(idx)+'Y_train.txt','w')
        for idx_train in range(1, K+1):
            if idx_train == idx:
                continue
            constructNetFromJson(path+'Fold'+str(idx_train)+'/dataFold'+str(idx_train), f_x=f_x, f_y = f_y)  # not reassign label
        f_x.close()
        f_y.close()

if __name__ == "__main__":
    # tokenvec.txt and toktypeDict.txt must be put in the data directory

    # Write X, Y for CodeChef Problem
    # problem: MNMX, FLOW016, SUBINC
    # problem = 'MNMX'
    # CodeChefExperiment(problem=problem)

    # Write X, Y for Virus Problem
    GraphJsonKFold(K=5, path=datapath)
    saveXYForKFold(K=5, path=datapath)
    print 'Done!!'
