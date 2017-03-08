# -*- coding: utf-8 -*-
import struct
import sys
import constructNetWork_CNNPair as TC

import MLP_DataIO
import params

sys.path.append('../nn')
datapath ='C:/Users/anhpv/Desktop/Long_Data/Bi_Corpus/indonesian-vietnamese/'

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

# initial the gradients

gradWeights = np.zeros_like(Weights)
gradBiases = np.zeros_like(Biases)

def InitByNodes(lan1_words, lan1_dict, lan2_words, lan2_dict):
    # ConstructTreeConvolution(lan1_words, lan1_dict, lan2_words, lan2_dict, numFea, numCon, numDis, numOut, \
    #                              lan1_Wconv_1, lan1_Wconv_2, lan1_Bconv, \
    #                              lan2_Wconv_1, lan2_Wconv_2, lan2_Bconv, \
    #                              Wlan1_dis, Wlan2_dis, Woutput, Bdis, Boutput
    #                              ):                         ):
    layers = TC.ConstructTreeConvolution(lan1_words, lan1_dict, lan2_words, lan2_dict, numFea, numCon, numDis, numOut, \
                                 lan1_Wconv_1, lan1_Wconv_2, lan1_Bconv, \
                                 lan2_Wconv_1, lan2_Wconv_2, lan2_Bconv, \
                                 Wlan1_dis, Wlan2_dis, Woutput, Bdis, Boutput
                                 )


    return layers

def InitNetbyText(text=''):

    phrases = text.split('|||')
    if len(phrases) !=2:
        print '\ndata sample error '+ text
    else:
        lan1_words = phrases[0].lstrip().rstrip().split(' ')
        lan2_words = phrases[1].lstrip().rstrip().split(' ')

        layers = InitByNodes(lan1_words = lan1_words, lan1_dict = lan1_dict, lan2_words = lan2_words, lan2_dict = lan2_dict)

    return layers
def ConstructNetworksFromFile(datafile='', classlabel =1, target=None):
    file = open(datafile, "r")
    idx =0
    for line in file:

        layers = InitNetbyText(text=line)
        #networks.append((layers, classlabel-1))

        # write networks
        out = open(target+str(idx), 'w')
        WriteNet(f= out, layers = layers)
        out.close()
        idx = idx +1
        if idx % 1000==0:
            print idx
        if idx >5:
            break

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
def testNet(text=''):
    # test net
    layers = InitNetbyText(text=text)
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



def writeXY():

    networks =[]
    #read pos nets

    # write network
    np.random.seed(314159)
    np.random.shuffle(networks)

    print 'networks =',len(networks)
    numTrain = int(.7 * len(networks))
    print 'numTrain : ', numTrain
    print 'numTest : ', len(networks) - numTrain

    f = file(datapath+'xy/' + 'data_train', 'wb')
    f_y = file(datapath+'xy/' + 'data_ytrain.txt', 'w')
    for i in xrange(0, numTrain):
        (net, ti) = networks[i]
        #write net
        WriteNet(f, net)
        # print ti
        f_y.write(str(ti) + '\n')
    f.close()
    f_y.close()

    f = file(datapath+'xy/' +  'data_CV', 'wb')
    f_y = file(datapath+'xy/' + 'data_yCV.txt', 'w')

    for i in xrange(numTrain, len(networks)):
        (net, ti) = networks[i]
        #write net
        WriteNet(f, net)
        #write ti
        f_y.write(str(ti) + '\n')
    f.close()
    f_y.close()

    f = file(datapath+'xy/' + 'data_test', 'wb')
    f_y = file(datapath+'xy/' + 'data_ytest.txt', 'w')

    for i in xrange(numTrain, numTrain+2):#len(networks)):
        (net, ti) = networks[i]
        # write net
        WriteNet(f, net)
        # write ti
        f_y.write(str(ti) + '\n')
    f.close()
    f_y.close()
if __name__ == "__main__":

    # # #positive file
    ConstructNetworksFromFile(datafile=datapath+'training-data.id-vi.positive.txt', classlabel=1, target=datapath+'/pos_net/')
    # #negative file
    ConstructNetworksFromFile(datafile=datapath + 'training-data.id-vi.negative.txt', classlabel=2, target=datapath+'/neg_net/')

    print 'Done!!'