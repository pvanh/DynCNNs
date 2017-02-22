import random
import struct

import constructNetWork_GCNN as TC
import Data_IO
import cPickle as p
from nn import serialize

import sys, os

sys.path.append('../nn')
datapath ='C:/Users/anhpv/Desktop/CFG_Virus/'

from InitParam import *

import numpy as np

sys.setrecursionlimit(1000000)

word_dict, vectors, numFea = Data_IO.LoadVocab(vocabfile=datapath + 'w2v_random.txt')
print 'Vocab length =', len(word_dict)
print 'Embedding size =', numFea

numDis = 10
numOut = 2

numCon =[2,3,4]

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
    layers = TC.ConstructTreeConvolution(graph, word_dict, numFea, numCon, numDis, numOut, \
                                 Wconv_root, Wconv_neighbor, Bconv, \
                                 Wdis, Woutput, Bdis, Boutput
                                 )


    return layers

# def ConstructNetworksFromFile(datafile='',targetdir='', prefix ='', classlabel =1, networks=[]):
#     file = open(datafile, "r")
#     idx =0
#     for line in file:
#
#         layers = InitNetbyText(text=line)
#         networks.append((layers, classlabel-1))
#         idx = idx +1
#         if idx % 1000==0:
#             print idx
        # if (idx>10):
        #     break

        # netfile =str(idx)
        # directory = targetdir + str(classlabel)
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # serialize.serialize(layers, directory + '/'+prefix+'_' + netfile)
def WriteNet(f =None, layers=None):
    num_lay = struct.pack('i', len(layers))
    if num_lay <= 2:
        print 'error'
    f.write(num_lay)

    num_con = 0

    #################################
    # preprocessing, compute some indexes
    for i, layer in enumerate(layers):
        layer.idx = i
        num_con += len(layer.connectUp)
        for (icon, con) in enumerate(layer.connectDown):
            con.ydownid = icon

    num_con = struct.pack('i', num_con)
    f.write(num_con)

    #################################
    # layers

    for layer in layers:
        # name
        # = struct.pack('s', layer.name )
        # numUnit
        tmp = struct.pack('i', layer.numUnit)
        f.write(tmp)
        # numUp
        tmp = struct.pack('i', len(layer.connectUp))
        f.write(tmp)
        # numDown
        tmp = struct.pack('i', len(layer.connectDown))
        f.write(tmp)

        if layer.layer_type == 'p':  # pooling
            if layer.poolType == 'max':
                tlayer = 'x'
            elif layer.poolType == 'sum':
                tlayer = 'u'
            tmp = struct.pack('c', tlayer)
            f.write(tmp)

        elif layer.layer_type == 'o':  # ordinary nodes

            if layer.act == 'embedding':
                tlayer = 'e'
            elif layer.act == 'autoencoding':
                tlayer = 'a'
            elif layer.act == 'convolution':
                tlayer = 'c'
            elif layer.act == 'combination':
                tlayer = 'b'
            elif layer.act == "ReLU":
                tlayer = 'r'
            elif layer.act == 'softmax':
                tlayer = 's'
            elif layer.act == 'hidden':
                tlayer = 'h'
            elif layer.act == 'recursive':
                tlayer = 'v'
            else:
                print "error"
                return layer
            tmp = struct.pack('c', tlayer)

            f.write(tmp)
            bidx = -1
            if layer.bidx != None:
                bidx = layer.bidx
                bidx = bidx[0]

            tmp = struct.pack('i', bidx)
            f.write(tmp)

    #########################
    # connections
    for layer in layers:
        for xupid, con in enumerate(layer.connectUp):
            # xlayer idx
            tmp = struct.pack('i', layer.idx)

            f.write(tmp)
            # ylayer idx
            tmp = struct.pack('i', con.ylayer.idx)
            f.write(tmp)
            # idx in x's connectUp
            tmp = struct.pack('i', xupid)
            f.write(tmp)
            # idx in y's connectDown
            tmp = struct.pack('i', con.ydownid)
            f.write(tmp)
            if con.ylayer.layer_type == 'p':
                Widx = -1
            else:
                Widx = con.Widx
                Widx = Widx[0]

            tmp = struct.pack('i', Widx)
            f.write(tmp)
            if Widx >= 0:
                tmp = struct.pack('f', con.Wcoef)
                f.write(tmp)
def testNet(filename):
    graph = Data_IO.getGraph(filename=filename)

    layers = InitByNodes(graph = graph , word_dict = word_dict)
    print 'Totally:', len(layers), 'layer(s)'
    for l in layers:
        if hasattr(l, 'bidx') and l.bidx is not None:
            print l.name, '\tBidx', l.bidx[0], '\tlen biases=', len(l.bidx)
        else:
            print l.name


        print "    Down:"
        for c in l.connectDown:
            if hasattr(c, 'Widx'):
                print "        ", c.xlayer.name, " -> ", '|', \
                    '(xnum= ', c.xnum, ', ynum= ', c.ynum, '), Wid = ', c.Widx[0]
            else:
                print "        ", c.xlayer.name, " -> ", '|', \
                    '(xnum= ', c.xnum, ', ynum= ', c.ynum, ')'




if __name__ == "__main__":

    g = Data_IO.getGraph(filename=datapath+'debug.exe_model.dot')
    g.show()
    testNet(filename=datapath+'debug.exe_model.dot')
    # # write network
    # np.random.seed(314159)
    # np.random.shuffle(networks)
    #
    # print 'networks =',len(networks)
    # numTrain = int(.7 * len(networks))
    # print 'numTrain : ', numTrain
    # print 'numTest : ', len(networks) - numTrain
    #
    # f = file(datapath+'xy/' + 'data_train', 'wb')
    # f_y = file(datapath+'xy/' + 'data_ytrain.txt', 'w')
    # for i in xrange(0, numTrain):
    #     (net, ti) = networks[i]
    #     #write net
    #     WriteNet(f, net)
    #     # print ti
    #     f_y.write(str(ti) + '\n')
    # f.close()
    # f_y.close()
    #
    # f = file(datapath+'xy/' +  'data_CV', 'wb')
    # f_y = file(datapath+'xy/' + 'data_yCV.txt', 'w')
    #
    # for i in xrange(numTrain, len(networks)):
    #     (net, ti) = networks[i]
    #     #write net
    #     WriteNet(f, net)
    #     #write ti
    #     f_y.write(str(ti) + '\n')
    # f.close()
    # f_y.close()
    #
    # f = file(datapath+'xy/' + 'data_test', 'wb')
    # f_y = file(datapath+'xy/' + 'data_ytest.txt', 'w')
    #
    # for i in xrange(numTrain, len(networks)):
    #     (net, ti) = networks[i]
    #     # write net
    #     WriteNet(f, net)
    #     # write ti
    #     f_y.write(str(ti) + '\n')
    # f.close()
    # f_y.close()



    print 'Done!!'
