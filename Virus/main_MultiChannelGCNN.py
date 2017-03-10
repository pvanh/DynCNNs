import json
import random
import struct

import constructNetWork_MultiChannelGCNN as TC
import GraphData_IO
import cPickle as p

from Graph import Graph
from nn import serialize

import sys, os
import gcnn_params as params

sys.path.append('../nn')
datapath = params.datapath

from InitParam import *

import numpy as np

sys.setrecursionlimit(1000000)

word_dict, vectors, numFea = GraphData_IO.LoadVocab(vocabfile=datapath + 'tokvec.txt')
print 'Vocab length =', len(word_dict)
print 'Embedding size =', numFea

numDis = params.numDis
numOut = params.numOut

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

Wconv_root=[]
Wconv_in =[]
Wconv_out =[]
Bconv =[]
# convolution layers
num_Pre = numFea
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



def InitByNodes(graph, word_dict):
    # Embedding ---> Conv1 ---> .... ---> Convn ---> Pooling ---> Fully-Connected ---> Output
    #(graph,word_dict,numView, numFea, numCon, numDis, numOut, \
                             # Wconv_root, Wconv_in, Wconv_out, Bconv, \
                             # Wdis, Woutput, Bdis, Boutput
                                 # ):

    # word_dict: [dict_view1, dict_view2]
    # numView, numFea, numCon, numDis, numOut, \
    # Wconv_root    [View1:[conv1, conv2, ...], View2:[conv1, conv2, ...]]
    # Wconv_income  [View1:[conv1, conv2, ...], View2:[conv1, conv2, ...]]
    # Wconv_out     [View1:[conv1, conv2, ...], View2:[conv1, conv2, ...]]
    # Bconv         [View1:[conv1, conv2, ...], View2:[conv1, conv2, ...]]
    # Wdis[pool_view1, pool_view2, ...---> Dis]
    layers = TC.ConstructGraphConvolution(graph, word_dict, numView, numFea, numCon, numDis, numOut, \
                                 Wconv_root, Wconv_in, Wconv_out, Bconv, \
                                 Wdis, Woutput, Bdis, Boutput
                                 )


    return layers


def constructNetFromJson(jsonFile='', labelclass=1): # 1: non-virus, 2: virus
    networks=[]
    with open(jsonFile, 'r') as f:
        jsonObjs = json.load(f)
        for obj in jsonObjs:
            graph = Graph.load(obj)
            g_net = InitByNodes(graph=graph, word_dict=word_dict)
            networks.append((g_net, labelclass-1))
    return networks

if __name__ == "__main__":

    datafiles =params.datafiles

    train_net =[]
    test_net =[]
    # training
    net = constructNetFromJson(jsonFile=datafiles['train_nonvirus'],labelclass=1)
    train_net.extend(net)
    net = constructNetFromJson(jsonFile=datafiles['train_virus'], labelclass=2)
    train_net.extend(net)
    #testing
    net = constructNetFromJson(jsonFile=datafiles['test_nonvirus'], labelclass=1)
    test_net.extend(net)
    net = constructNetFromJson(jsonFile=datafiles['test_virus'], labelclass=2)
    test_net.extend(net)
    # # write network
    # np.random.seed(314159)
    # np.random.shuffle(networks)
    #
    # print 'networks =',len(networks)
    # numTrain = int(.7 * len(networks))
    print 'numTrain : ', len(train_net)
    print 'numTest : ', len(test_net)
    np.random.shuffle(train_net)

    f = file(datapath+'xy/' + 'data_train', 'wb')
    f_y = file(datapath+'xy/' + 'data_ytrain.txt', 'w')
    for i in xrange(0, len(train_net)):
        (net, ti) = train_net[i]
        #write net
        GraphData_IO.WriteNet(f, net)
        # print ti
        f_y.write(str(ti) + '\n')
    f.close()
    f_y.close()

    f = file(datapath+'xy/' +  'data_test', 'wb')
    f_y = file(datapath+'xy/' + 'data_ytest.txt', 'w')
    for i in xrange(0, len(test_net)):
        (net, ti) = test_net[i]
        #write net
        GraphData_IO.WriteNet(f, net)
        #write ti
        f_y.write(str(ti) + '\n')
    f.close()
    f_y.close()


    print 'Done!!'
