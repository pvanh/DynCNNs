
import GraphData_IO

import write_param
import sys
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

print Weights[1], Weights[2], Weights[3], Weights[4], Weights[5]
print 'numDis', numDis
print 'numCon', numCon
print 'Weights', len(Weights)
print 'Bias', len(Biases)
# dwadwad
# 17940
# 1544
#
write_param.write_binary('../paramTest_GCNN', Weights, Biases)
print 'Done!'