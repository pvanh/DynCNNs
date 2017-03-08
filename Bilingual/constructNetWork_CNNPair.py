import sys

sys.path.append('../nn')
sys.path.append('../')
import Layers as Lay, Connections as Con, Activation


##############################
# hyperparam
# dict_bid: the start index of the dictionary
def senNet(words , word_dict, dict_bid,\
                       numFea, numCon,
                       Wconv_1, Wconv_2, Bconv):
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
def ConstructTreeConvolution(lan1_words, lan1_dict, lan2_words, lan2_dict, numFea, numCon,numDis, numOut, \
                             lan1_Wconv_1, lan1_Wconv_2, lan1_Bconv, \
                             lan2_Wconv_1, lan2_Wconv_2, lan2_Bconv, \
                             Wlan1_dis, Wlan2_dis, Woutput, Bdis, Boutput
                             ):
    # phrase_1, phrase_2, word_dict, numFea, numLeft, numRight, numJoint, numDis, numOut, \
    # Wleft, Wright, Bleft, Bright,
    # Wjoint_left, Wjoint_right, Bjoint,
    # Wdis, Wout, Bdis, Bout                          ):
    # w1 - ---> CNNs |
    # w2 - ---> CNNs | Lan1_Sen  --->    S1|
    # w3 - ---> CNNs |                     |
    #                                      | ---> Fully - Connected ---> out
    # w1 - ---> CNNs |                     |
    # w2 - ---> CNNs | Lan2_Sen --->     S2|
    # w3 - ---> CNNs |

    pool_lan1 = Lay.PoolLayer('pool_lan1', numCon[-1])
    pool_lan2 = Lay.PoolLayer('pool_lan2', numCon[-1])

    layers=[]

    #construct net for sentence - language 1

    lan1_sen_layers, convOut_layer = senNet(lan1_words, word_dict=lan1_dict,dict_bid=0,
                                       numFea = numFea,numCon=numCon,Wconv_1=lan1_Wconv_1, Wconv_2=lan1_Wconv_2, Bconv=lan1_Bconv)

    layers.extend(lan1_sen_layers)
    for layer in convOut_layer:
        poolCon = Con.PoolConnection(layer, pool_lan1)


    dict_bid = len(lan1_dict)*numFea
    lan2_sen_layers, convOut_layer = senNet(lan2_words, word_dict=lan2_dict, dict_bid= dict_bid,
                                                numFea=numFea, numCon=numCon, Wconv_1=lan2_Wconv_1,
                                                Wconv_2=lan2_Wconv_2, Bconv=lan2_Bconv)


    layers.extend(lan2_sen_layers)
    for layer in convOut_layer:
        poolCon = Con.PoolConnection(layer, pool_lan2)

    layers.append(pool_lan1)
    layers.append(pool_lan2)

    #create discriminative and output layers
    discriminative = Lay.layer('discriminative', Bdis, numDis)
    discriminative.act = 'hidden'
    output = Lay.layer('outputlayer', Boutput, numOut)
    output.act = 'softmax'
    # connect pooling layers ---> discriminative layer
    con = Con.connection(pool_lan1, discriminative, numCon[-1], numDis, Wlan1_dis)
    con = Con.connection(pool_lan2, discriminative, numCon[-1], numDis, Wlan2_dis)

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