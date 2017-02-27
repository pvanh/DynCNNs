import cPickle as p
import struct
import sys
import constructNetWork_CNN as TC

import MLP_DataIO
import Data_IO
import write_param

sys.path.append('../nn')
datapath ='C:/Users/anhpv/Desktop/COLIEE/data/'


from InitParam import *

import gl
from nn import Token
import numpy as np


word_dict, vectors, numFea = MLP_DataIO.LoadVocab(vocabfile=datapath + 'pretrained/text-format-glove/rre.w2v50.txt')
print 'Vocab length =', len(word_dict)
print 'Embedding size =', numFea

numDis = 300
numOut = 2

numCon =[300,300,300]

np.random.seed(314)

preBword = vectors # numFea*len(Vocab) # word vectors
print '(Words = ', len(word_dict), ') * (Size =', numFea,') = ', len(preBword)
# Initialize weights and biases
Weights = np.array([])
Biases = np.array([])

# word/symbol/token embedding
Biases, BwordIdx = InitParam(Biases, newWeights=preBword)
Wconv_1=[]
Wconv_2 = []
Bconv =[]
# convolution layers
num_Pre = numFea
for c in range(len(numCon)):
    Weights, w = InitParam(Weights, num=num_Pre * numCon[c])
    Wconv_1.append(w)

    Weights, w = InitParam(Weights, num=num_Pre * numCon[c])
    Wconv_2.append(w)

    Biases, b = InitParam(Biases, num=numCon[c])
    Bconv.append(b)

    num_Pre = numCon[c]

# discriminative layer
Weights, Wques_dis = InitParam(Weights, num=num_Pre * numDis)
Weights, Warticle_dis = InitParam(Weights, num=num_Pre * numDis)
Biases, Bdis = InitParam(Biases, num=numDis)

# output layer
Weights, Woutput = InitParam(Weights, num=numDis * numOut, upper=.0002, lower=-.0002)
Biases, Boutput = InitParam(Biases, newWeights=np.zeros((numOut, 1)))

Weights = Weights.reshape((-1, 1))
Biases = Biases.reshape((-1, 1))

print 'num of biases = ', len(Biases)
print 'num of weights = ', len(Weights)


write_param.write_binary('../paramTest_CNN', Weights, Biases)
print 'Done!'