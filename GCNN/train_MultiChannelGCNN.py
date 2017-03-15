import GraphData_IO

import write_param

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
print 'numCon', numCon
print 'numDis', numDis
print 'numOut', numOut
print 'Weights', len(Weights)
print 'Bias', len(Biases)
# dwadwad
# 17940
# 1544
#
write_param.write_binary(datapath+'paramTest_ASTGraph_V'+str(numView), Weights, Biases)
print 'Done!'