import random
import codecs
import sys, os
import numpy as np


def LoadVocabFromTrain_TestFiles(vocabfiles):

    dict_lan1={}
    dict_lan2 = {}

    idx_l1 =0
    idx_l2 = 0
    # get word from vocab files
    cout =0
    for datafile in vocabfiles:
        file = codecs.open(datafile,encoding='utf-8', mode='r')
        for line in file:
            # sentence language 1 ||| sentence language 2
            phrases = line.split('|||')
            phrase_1 = phrases[0].lstrip().rstrip().split(' ')
            phrase_2 = phrases[1].lstrip().rstrip().split(' ')

            for word in phrase_1:
                if word in dict_lan1.keys():
                    continue

                dict_lan1[word] = idx_l1
                idx_l1 = idx_l1 +1

            for word in phrase_2:
                if word in dict_lan2.keys():
                    continue

                dict_lan2[word] = idx_l2
                idx_l2 = idx_l2 + 1
        file.close()

    return dict_lan1, dict_lan2

def WriteWord_Vec(word, vecsize ,file):
    file.write(word + ' ')
    vec = [random.uniform(-1, 1) for v in xrange(vecsize)]
    vec = [str(i) for i in vec]
    file.write(' '.join(vec))
    file.write('\n')
def GenerateLang_Dict(lan1='ind', lan2='vn'): # create dictionaries
    datapath ='C:/Users/anhpv/Desktop/Long_Data/Bi_Corpus/indonesian-vietnamese/'
    dict_lan1, dict_lan2 = LoadVocabFromTrain_TestFiles([datapath+'training-data.id-vi.negative.txt', datapath+'training-data.id-vi.positive.txt'])

    vecsize = 50
    #write dict 1
    file = codecs.open(datapath+'w2vrandom_'+ lan1, encoding='utf-8', mode='w')
    file.write(str(len(dict_lan1)+1)+' '+str(vecsize)+'\n')

    #insert unknow work
    WriteWord_Vec(word='__unknown__', vecsize= vecsize, file = file)
    for w in dict_lan1:
        WriteWord_Vec(word=w, vecsize= vecsize, file= file)
    file.close()


    # write dict 2
    file = codecs.open(datapath + 'w2vrandom_' + lan2, encoding='utf-8', mode='w')
    file.write(str(len(dict_lan2)+1) + ' ' + str(vecsize) + '\n')

    # insert unknow work
    WriteWord_Vec(word='__unknown__', vecsize=vecsize, file=file)
    for w in dict_lan2:
        WriteWord_Vec(word=w, vecsize=vecsize, file=file)
    file.close()

def writeXY(posDir, negDir, xyDir):

    networks = []
    # read pos nets
    for onefile in os.listdir(posDir):
        f = open(posDir + onefile,'r')
        networks.append((f.read(),0))
        f.close()
    #read neg nets
    for onefile in os.listdir(negDir):
        f = open(posDir + onefile,'r')
        networks.append((f.read(),1))
        f.close()

    print len(networks)
    # write network
    np.random.seed(314159)
    np.random.shuffle(networks)

    print 'networks =', len(networks)
    numTrain = int(.7 * len(networks))
    print 'numTrain : ', numTrain
    print 'numTest : ', len(networks) - numTrain

    f = file(xyDir + 'data_train', 'wb')
    f_y = file(xyDir + 'data_ytrain.txt', 'w')
    for i in xrange(0, numTrain):
        (net, ti) = networks[i]
        # write net
        f.write(net)
        # print ti
        f_y.write(str(ti) + '\n')
    f.close()
    f_y.close()

    f = file(xyDir + 'data_CV', 'wb')
    f_y = file(xyDir + 'data_yCV.txt', 'w')

    for i in xrange(numTrain, len(networks)):
        (net, ti) = networks[i]
        # write net
        f.write(net)
        # write ti
        f_y.write(str(ti) + '\n')
    f.close()
    f_y.close()

    f = file(xyDir + 'data_test', 'wb')
    f_y = file(xyDir + 'data_ytest.txt', 'w')

    for i in xrange(numTrain, numTrain + 2):  # len(networks)):
        (net, ti) = networks[i]
        # write net
        f.write(net)
        # write ti
        f_y.write(str(ti) + '\n')
    f.close()
    f_y.close()

datapath ='C:/Users/anhpv/Desktop/Long_Data/Bi_Corpus/indonesian-vietnamese/'
writeXY(posDir= datapath+'/pos_net/', negDir=datapath+'/neg_net/', xyDir=datapath+'xy/')