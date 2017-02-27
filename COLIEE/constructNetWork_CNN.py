import sys

sys.path.append('../nn')
sys.path.append('../')
import Layers as Lay, Connections as Con, Activation


##############################
# hyperparam
# dict_bid: the start index of the dictionary
def senNet(sen , word_dict, dict_bid,\
                       numFea, numCon,
                       Wconv_1, Wconv_2, Bconv):
    words = sen.getWords()
    numWords = len(words)

    layers = [None] * numWords

    # construct layers for each node
    for idx in xrange(numWords):
        word = words[idx]
        if word in word_dict:
            bidx = dict_bid + word_dict[word] * numFea
        else:
            #print word
            bidx =dict_bid + 0
        layers[idx] = Lay.layer('vec_' + str(idx) + '_' + word, \
                                range(bidx, bidx + numFea), \
                                numFea
                                )
        layers[idx].act = 'embedding'

    pre_layer ={}
    for idx in xrange(numWords):
        pre_layer[idx] = layers[idx]

    num_Pre = numFea # the size of previous layers
    for c in xrange(len(numCon)): # convolutional layers
        current_layer ={} # current layer
        for idx in xrange(numWords):
            idx1 = idx # word 1 in sliding window
            idx2 = idx1+1 # word 2

            conLayer = Lay.layer('Convolve'+str(c)+'_' + words[idx], Bconv[c], numCon[c])
            conLayer.act = 'convolution'
            layers.append(conLayer)
            current_layer[idx] = conLayer
            # add word 1 connection
            word1Con = Con.connection(pre_layer[idx1], conLayer, num_Pre, numCon[c], Wconv_1[c])
            # add word 2 connection
            if idx2< numWords:
                word2Con = Con.connection(pre_layer[idx2], conLayer, num_Pre, numCon[c], Wconv_2[c])
        num_Pre = numCon[c]
        pre_layer = current_layer
    return layers, pre_layer.values()
def ConstructTreeConvolution(ques, rel_articles,word_dict, numFea, numCon,numDis, numOut, \
                             Wconv_1, Wconv_2, Bconv, \
                             Wques_dis, Warticle_dis, Woutput, Bdis, Boutput
                             ):
    # phrase_1, phrase_2, word_dict, numFea, numLeft, numRight, numJoint, numDis, numOut, \
    # Wleft, Wright, Bleft, Bright,
    # Wjoint_left, Wjoint_right, Bjoint,
    # Wdis, Wout, Bdis, Bout                          ):
    # w1 - ---> CNNs |
    # w2 - ---> CNNs | Ques  ---> Pooling Q|
    # w3 - ---> CNNs |                     |
    #                                 | ---> Fully - Connected ---> out
    # w1 - ---> CNNs |                     |
    # w2 - ---> CNNs | right ---> Pooling A|
    # w3 - ---> CNNs |

    pool_ques = Lay.PoolLayer('pool_ques', numCon[-1])
    pool_articles = Lay.PoolLayer('pool_articles', numCon[-1])

    layers=[]


    ques_sen = ques.sen

    sen_layers, convOut_layer = senNet(ques_sen, word_dict=word_dict,dict_bid=0,
                                       numFea = numFea,numCon=numCon,Wconv_1=Wconv_1, Wconv_2=Wconv_2, Bconv=Bconv)

    layers.extend(sen_layers)
    for layer in convOut_layer:
        poolCon = Con.PoolConnection(layer, pool_ques)

    # layers for articles
    for article in rel_articles:
        sens = article.sens
        for sen in sens:
            sen_layers, convOut_layer = senNet(sen, word_dict=word_dict, dict_bid=0,
                                               numFea=numFea, numCon=numCon, Wconv_1=Wconv_1, Wconv_2=Wconv_2,
                                               Bconv=Bconv)
            layers.extend(sen_layers)
            for layer in convOut_layer:
                poolCon = Con.PoolConnection(layer, pool_articles)

    layers.append(pool_ques)
    layers.append(pool_articles)

    #create discriminative and output layers
    discriminative = Lay.layer('discriminative', Bdis, numDis)
    discriminative.act = 'hidden'
    output = Lay.layer('outputlayer', Boutput, numOut)
    output.act = 'softmax'
    # connect pooling layers ---> discriminative layer
    con = Con.connection(pool_ques, discriminative, numCon[-1], numDis, Wques_dis)
    con = Con.connection(pool_ques, discriminative, numCon[-1], numDis, Warticle_dis)

    outcon = Con.connection(discriminative, output, numDis, numOut, Woutput)
    if numOut > 1:
        output._activate = Activation.softmax
        output._activatePrime = None
    layers.append(discriminative)
    layers.append(output)
    # add successive connections
    numlayers = len(layers)
    for idx in xrange(numlayers):
        if idx > 0:
            layers[idx].successiveLower = layers[idx - 1]
        if idx < numlayers - 1:
            layers[idx].successiveUpper = layers[idx + 1]
    return layers