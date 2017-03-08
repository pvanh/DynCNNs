import json
import random
import struct

import constructNetWork_GCNN as TC
import Data_IO
import cPickle as p

import Graph
from nn import serialize

import sys, os
import gcnn_params as params

sys.path.append('../nn')
datapath = params.datapath

from InitParam import *

import numpy as np

sys.setrecursionlimit(1000000)

word_dict, vectors, numFea = Data_IO.LoadVocab(vocabfile=datapath + 'tokvec.txt')
print 'Vocab length =', len(word_dict)
print 'Embedding size =', numFea

numDis = params.numDis
numOut = params.numOut

numCon = params.numCon

np.random.seed(314)

preBword = vectors # numFea*len(Vocab) # word vectors
print '(Words = ', len(word_dict), ') * (Size =', numFea,') = ', len(preBword)
# Initialize weights and biases
Weights = np.array([])
Biases = np.array([])

# word/symbol/token embedding
Biases, BwordIdx = InitParam(Biases, newWeights=preBword)
Wconv_root=[]
Wconv_neighbor = []
Bconv =[]
# convolution layers
num_Pre = numFea
for c in range(len(numCon)):
    Weights, w = InitParam(Weights, num=num_Pre * numCon[c])
    Wconv_root.append(w)

    Weights, w = InitParam(Weights, num=num_Pre * numCon[c])
    Wconv_neighbor.append(w)

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
    # ConstructTreeConvolution(graph, word_dict, numFea, numCon, numDis, numOut, \
    #                              Wconv_root, Wconv_neighbor, Bconv, \
    #                              Wdis, Woutput, Bdis, Boutput
    #                              ):
    layers = TC.ConstructGraphConvolution(graph, word_dict, numFea, numCon, numDis, numOut, \
                                 Wconv_root, Wconv_neighbor, Bconv, \
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

    # g = Data_IO.getGraph(filename=datapath+'test.dot')
    # g.show()
    # testNet(filename=datapath+'test.dot')

    datafiles = params.datafiles

    train_net =[]
    test_net =[]
    # training
    net = constructNetFromJson(jsonFile=datafiles['train_nonvirus'],labelclass=1)
    train_net.append(net)
    net = constructNetFromJson(jsonFile=datafiles['train_virus'], labelclass=2)
    train_net.append(net)
    #testing
    net = constructNetFromJson(jsonFile=datafiles['test_nonvirus'], labelclass=1)
    test_net.append(net)
    net = constructNetFromJson(jsonFile=datafiles['test_virus'], labelclass=2)
    test_net.append(net)
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
        Data_IO.WriteNet(f, net)
        # print ti
        f_y.write(str(ti) + '\n')
    f.close()
    f_y.close()

    f = file(datapath+'xy/' +  'data_test', 'wb')
    f_y = file(datapath+'xy/' + 'data_ytest.txt', 'w')
    for i in xrange(0, len(test_net)):
        (net, ti) = test_net[i]
        #write net
        Data_IO.WriteNet(f, net)
        #write ti
        f_y.write(str(ti) + '\n')
    f.close()
    f_y.close()





    print 'Done!!'
