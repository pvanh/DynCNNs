import sys

sys.path.append('../nn')
sys.path.append('../')
import Layers as Lay, Connections as Con, Activation

import gcnn_params
##############################
# hyperparam
toktypeDict = gcnn_params.toktypeDict
class Vinfo:
    def __init__(self, id, data = None,income=None, outgo =None):
        if income is None:
            self.income=[]
        else:
            self.income = income
        if outgo is None:
            self.outgo = []
        else:
            self.outgo = outgo
        if data is None:
            self.data=[]
        else:
            self.data =data
        self.id = id
    def inDegree(self):
        return len(self.income)
    def outDegree(self):
        return len(self.outgo)
    def show(self, buf = sys.stdout):
        print 'id = ',str(self.id)
        print  'data=',self.data
        print  'incoming', self.income
        print  str(self.numIn())
        print 'outgoing', self.outgo
        print str(self.numOut()),'\n\n'


def ConstructGraphConvolution(graph,word_dict,numView, numFea, numCon, numDis, numOut, \
                             Wconv_root, Wconv_neighbor, Bconv, \
                             Wdis, Woutput, Bdis, Boutput
                             ):
    #word_dict: [dict_view1, dict_view2]
    #numView, numFea, numCon, numDis, numOut, \
    #Wconv_root    [View1:[conv1, conv2, ...], View2:[conv1, conv2, ...]]
    #Wconv_income  [View1:[conv1, conv2, ...], View2:[conv1, conv2, ...]]
    #Wconv_out     [View1:[conv1, conv2, ...], View2:[conv1, conv2, ...]]
    #Bconv         [View1:[conv1, conv2, ...], View2:[conv1, conv2, ...]]
    #Wdis[pool_view1, pool_view2, ...---> Dis]

    vertexes = graph.getVertexes() # dict of nodes: id - node
    numVertexes = len(vertexes)

    edges = graph.getEdges() # list of edges[(id1--->id2)]
    numEdges = len(edges)

    vertexes_info ={}
    # get information of vertexes
    for idx in range(numVertexes):
        v= vertexes[idx]
        vinfor =Vinfo(id=v.id, data=v.getViews(toktypeDict))
        for v1, v2 in edges: # outgoing edge
            if idx== v1:
                vinfor.outgo.append(v2)
            if idx == v2: # incoming edge
                vinfor.income.append(v1)
        vertexes_info[idx] = vinfor
    # sum of indegree, sum of out degree of neighbors
    n_degrees ={}
    for idx in vertexes_info:
        sumin =0
        sumout =0
        vinfor = vertexes_info[idx]
        for nid in vinfor.outgo:
            neighbor = vertexes_info[nid]
            sumin += neighbor.inDegree()
            sumout += neighbor.outDegree()

        n_degrees[vinfor.id] =(sumin, sumout)

    layers=[]
    pool = [None]*numView # pooling layers

    for view in range(0, numView):
        view_layers = [None] * numVertexes
        # weights for the current view
        view_wroot = Wconv_root[view]
        view_wneighbor = Wconv_neighbor[view]
        view_bconv = Bconv[view]
        # construct the embedding layer for each vertex
        for idx in xrange(numVertexes):
            # get vertex
            vinfor = vertexes_info[idx]
            token = vinfor.data[view] # get the token of the current view
            if token in word_dict:
                bidx = word_dict[token] * numFea
            else:
                bidx =0
            view_layers[idx] = Lay.layer('vec_' + str(idx) + '_' + token, \
                                    range(bidx, bidx + numFea), \
                                    numFea
                                    )
            view_layers[idx].act = 'embedding'

        pre_layer ={}
        for idx in xrange(numVertexes):
            pre_layer[idx] = view_layers[idx]

        num_Pre = numFea # the size of previous layers
        for c in xrange(len(numCon)): # convolutional layers
            current_layer ={} # current layer
            for idx in xrange (numVertexes):
                # current vertex V[idx]
                vinfor = vertexes_info[idx]
                token = vinfor.data[view]  # get the token

                conLayer = Lay.layer('Convolve'+str(c)+'_' + token+'_V'+str(view+1), view_bconv[c], numCon[c])
                conLayer.act = 'convolution'
                view_layers.append(conLayer)
                current_layer[idx] = conLayer
                # add root connection
                rootCon = Con.connection(pre_layer[idx], conLayer, num_Pre, numCon[c], view_wroot[c])
                # add incoming connections
                n_sumin, n_sumout = n_degrees[idx]
                n_sumDegree = n_sumin + n_sumout
                if n_sumDegree==0:
                    n_sumDegree = 1

                for n in vinfor.outgo:
                    # print token, vertexes_info[n].inDegree(), ',', vertexes_info[n].outDegree(), ',', n_sumDegree
                    Wcoef = 1.0* (vertexes_info[n].inDegree()+vertexes_info[n].outDegree())/ n_sumDegree
                    if Wcoef !=0:
                        neighborCon = Con.connection(pre_layer[n], conLayer, num_Pre, numCon[c], view_wneighbor[c],Wcoef = Wcoef)

            num_Pre = numCon[c]
            pre_layer = current_layer

        pool[view] = Lay.PoolLayer('pooling'+str(view+1), numCon[-1])
        view_layers.append(pool[view])
        # connect from convolution ---> pooling
        for key in pre_layer:
            poolCon = Con.PoolConnection(pre_layer[key], pool[view])
        # add view layers to layers
        layers.extend(view_layers)

    # discriminative layer
    discriminative = Lay.layer('discriminative', Bdis, numDis)
    discriminative.act = 'hidden'
    # pool ---> discriminative
    for idx in range(0, numView):
        con = Con.connection(pool[idx], discriminative, numCon[-1], numDis, Wdis[idx])

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
