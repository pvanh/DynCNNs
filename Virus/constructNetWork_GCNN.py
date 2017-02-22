import sys

sys.path.append('../nn')
sys.path.append('../')
import Layers as Lay, Connections as Con, Activation


##############################
# hyperparam
class info:
    parent = None
    childrenList = None

    def __init__(self, parent=None):
        self.parent = parent
        self.childrenList = []


def ConstructTreeConvolution(graph,word_dict, numFea, numCon, numDis, numOut, \
                             Wconv_root, Wconv_neighbor, Bconv, \
                             Wdis, Woutput, Bdis, Boutput
                             ):
    # nodes
    # word_dict: list of tokens
    # numFea: # of the word/symbol feature size
    # numCon: # of the convolution size
    # Wleft:  left  weights of continous binary tree autoencoder
    # Wright: right weights of continous binary tree autoencoder
    # Bconstruct: the biase for the autoencoder
    # Wcomb_ae, Wcomb_orig: the weights for the combination of
    #                       autoencoder and the original vector
    #                       (no biase for this sate)
    # Wconv_root, Wconv_left, Wconv_right, Bconv: the weights for covolution
    # Bconv: Biases for covolution

    vertexes = graph.getVertexes() # dict of node: id - node
    numVertexes = len(vertexes)

    edges = graph.getEdges() # list of edge [(id1--->id2)]
    numEdges = len(edges)
    neighbors ={}
    for idx in range(numVertexes):
        n =[]
        for v1, v2 in edges:
            if v1 == idx:
                n.append(v2)
        neighbors[idx] = n

    layers = [None] * numVertexes

    # construct layers for each node
    for idx in xrange(numVertexes):
        v = vertexes[idx]
        if v.value in word_dict:
            bidx = word_dict[v.value] * numFea
        else:
            bidx =0
        layers[idx] = Lay.layer('vec_' + str(idx) + '_' + v.value, \
                                range(bidx, bidx + numFea), \
                                numFea
                                )
        layers[idx].act = 'embedding'

    pre_layer ={}
    for idx in xrange(numVertexes):
        pre_layer[idx] = layers[idx]

    num_Pre = numFea # the size of previous layers
    for c in xrange(len(numCon)): # convolutional layers
        current_layer ={} # current layer
        for idx in xrange (numVertexes):
            # current vertex V[idx]
            v = vertexes[idx]
            conLayer = Lay.layer('Convolve'+str(c)+'_' + v.value, Bconv[c], numCon[c])
            conLayer.act = 'convolution'
            layers.append(conLayer)
            current_layer[idx] = conLayer
            # add root connection
            rootCon = Con.connection(pre_layer[idx], conLayer, num_Pre, numCon[c], Wconv_root[c])
            # add neighbor connections
            for n in neighbors[idx]:
                neighborCon = Con.connection(pre_layer[n], conLayer, num_Pre, numCon[c], Wconv_neighbor[c])

        num_Pre = numCon[c]
        pre_layer = current_layer

    pool = Lay.PoolLayer('pooling', numCon)
    layers.append(pool)
    # connect from convolution ---> pooling
    for key in pre_layer:
        poolCon = Con.PoolConnection(pre_layer[key], pool)

    # discriminative layer
    discriminative = Lay.layer('discriminative', Bdis, numDis)
    discriminative.act = 'hidden'
    # pool ---> discriminative
    con = Con.connection(pool, discriminative, numCon[-1], numDis, Wdis)

    output = Lay.layer('outputlayer', Boutput, numOut)
    output.act = 'softmax'
    # discriminative ---> output
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
