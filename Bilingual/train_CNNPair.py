import sys

import MLP_DataIO
import write_param
import params

sys.path.append('../nn')
datapath ='C:/Users/anhpv/Desktop/Long_Data/Bi_Corpus/malaysian-vietnamese/'

print datapath

from InitParam import *

import numpy as np

lan1_dict, lan1_vectors, numFea = MLP_DataIO.LoadVocab(params.lan1_dictfile)
print 'Vocab 1 length =', len(lan1_dict)
print 'Embedding size 1 =', len(lan1_vectors)

lan2_dict, lan2_vectors, numFea = MLP_DataIO.LoadVocab(params.lan2_dictfile)
print 'Vocab 2 length =', len(lan2_dict)
print 'Embedding size 2 =', len(lan2_vectors)
print 'Vector size = ', numFea

numCon = params.numCon
numDis = params.numDis
numOut = params.numOut

np.random.seed(314)

preBword = [] # numFea*len(Vocab) # word vectors
preBword.extend(lan1_vectors)
preBword.extend(lan2_vectors)
print 'Embedding size = ', len(preBword),'\n\n'
# Initialize weights and biases
Weights = np.array([])
Biases = np.array([])

# word/symbol/token embedding
Biases, BwordIdx = InitParam(Biases, newWeights=preBword)
lan1_Wconv_1=[]
lan1_Wconv_2 = []
lan1_Bconv =[]

lan2_Wconv_1=[]
lan2_Wconv_2 = []
lan2_Bconv =[]

# convolution layers for language 1
num_Pre = numFea
for c in range(len(numCon)):
    Weights, w = InitParam(Weights, num=num_Pre * numCon[c])
    lan1_Wconv_1.append(w)

    Weights, w = InitParam(Weights, num=num_Pre * numCon[c])
    lan1_Wconv_2.append(w)

    Biases, b = InitParam(Biases, num=numCon[c])
    lan1_Bconv.append(b)

    num_Pre = numCon[c]

# convolution layers for language 2
num_Pre = numFea
for c in range(len(numCon)):
    Weights, w = InitParam(Weights, num=num_Pre * numCon[c])
    lan2_Wconv_1.append(w)

    Weights, w = InitParam(Weights, num=num_Pre * numCon[c])
    lan2_Wconv_2.append(w)

    Biases, b = InitParam(Biases, num=numCon[c])
    lan2_Bconv.append(b)

    num_Pre = numCon[c]

# discriminative layer
Weights, Wlan1_dis = InitParam(Weights, num=num_Pre * numDis)
Weights, Wlan2_dis = InitParam(Weights, num=num_Pre * numDis)
Biases, Bdis = InitParam(Biases, num=numDis)

# output layer
Weights, Woutput = InitParam(Weights, num=numDis * numOut, upper=.0002, lower=-.0002)
Biases, Boutput = InitParam(Biases, newWeights=np.zeros((numOut, 1)))

Weights = Weights.reshape((-1, 1))
Biases = Biases.reshape((-1, 1))

print 'num of biases = ', len(Biases)
print 'num of weights = ', len(Weights)


write_param.write_binary('../paramTest_MS_VN', Weights, Biases)
print 'Done!'