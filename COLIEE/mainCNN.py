import cPickle as p
import struct
import sys
import constructNetWork_CNN as TC

import MLP_DataIO
import Data_IO

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

# initial the gradients

gradWeights = np.zeros_like(Weights)
gradBiases = np.zeros_like(Biases)

def InitByNodes(ques, articles, word_dict):
    # ConstructTreeConvolution(ques, rel_articles, word_dict, numFea, numCon, numDis, numOut, \
    #                              Wconv_1, Wconv_2, Bconv, \
    #                              Wques_dis, Warticle_dis, Woutput, Bdis, Boutput
    #                              ):
    layers = TC.ConstructTreeConvolution(ques, articles, word_dict, numFea, numCon, numDis, numOut, \
                                 Wconv_1, Wconv_2, Bconv, \
                                 Wques_dis, Warticle_dis, Woutput, Bdis, Boutput
                                 )


    return layers

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
def testNet(ques, article):
    layers = InitByNodes(ques, article, word_dict)
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

    datapath = 'C:/Users/anhpv/Desktop/COLIEE/data/'
    # questions
    filename = datapath + 'training-data-parsed/gold-all.json'
    questions = Data_IO.getQuestions(filename=filename)
    # articles
    filename = datapath + 'civil-code-parsed/ref/all_articles_parsed.json'
    articles = Data_IO.getArticles(filename=filename)

    train_nets =[]
    test_nets =[]

    for q in questions:
        rel_A =[]
        for id_a in q.article_ids:
            rel_A.append(articles[id_a])
        net = InitByNodes(ques=q, articles= rel_A, word_dict=word_dict)

        print q.label
        label = 1
        if q.label:
            label = 0 # 0: True, 1: False

        if q.id.startswith('H27'):
            test_nets.append((net, label))
        else:
            train_nets.append((net,label))


    # write network

    print 'networks =',len(train_nets)+len(test_nets)
    print 'numTrain : ', len(train_nets)
    print 'numTest : ', len(test_nets)

    f = file(datapath+'xy/' + 'data_train', 'wb')
    f_y = file(datapath+'xy/' + 'data_ytrain.txt', 'w')
    for i in xrange(0, len(train_nets)):
        (net, ti) = train_nets[i]
        #write net
        WriteNet(f, net)
        # print ti
        f_y.write(str(ti) + '\n')
    f.close()
    f_y.close()

    f = file(datapath+'xy/' +  'data_CV', 'wb')
    f_y = file(datapath+'xy/' + 'data_yCV.txt', 'w')

    for i in xrange(0, len(test_nets)):
        (net, ti) = test_nets[i]
        #write net
        WriteNet(f, net)
        #write ti
        f_y.write(str(ti) + '\n')
    f.close()
    f_y.close()

    f = file(datapath+'xy/' + 'data_test', 'wb')
    f_y = file(datapath+'xy/' + 'data_ytest.txt', 'w')

    for i in xrange(0, len(test_nets)):
        (net, ti) = test_nets[i]
        # write net
        WriteNet(f, net)
        # write ti
        f_y.write(str(ti) + '\n')
    f.close()
    f_y.close()



    print 'Done!!'