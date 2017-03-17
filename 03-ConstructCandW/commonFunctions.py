import struct

import gl
from nn import Token


def computeLeafNum(root, nodes, depth=0):
    if len(root.children) == 0:
        root.leafNum = 1
        root.childrenNum = 1
        return 1, 1, depth  # leafNum, childrenNum
    root.allLeafNum = 0
    avgdepth = 0.0
    for child in root.children:
        leafNum, childrenNum, childAvgDepth = computeLeafNum(nodes[child], nodes, depth + 1)
        root.leafNum += leafNum
        root.childrenNum += childrenNum
        avgdepth += childAvgDepth * leafNum
    avgdepth /= root.leafNum
    root.childrenNum += 1
    return root.leafNum, root.childrenNum, avgdepth

def ConstructNodes(ast, name, parent, pos, nodes, leafs, tokenMap):
    # global tmpCnt
    if name is None:
        name = ast.content

    Node = Token.token(name, gl.numFea * tokenMap[name], \
                       parent, pos)
    if len(ast.children) == 0:
        leafs.append(Node)
    else:
        nodes.append(Node)
    # print nodes[0].word
    curid = len(nodes)
    for idx, child in enumerate(ast.children):
        ConstructNodes(child, None, curid, idx, nodes, leafs, tokenMap)

def AdjustOrder(nodes):
    nodes.reverse()
    length = len(nodes)
    for node in nodes:
        if node.parent != None:
            node.parent = length - node.parent

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
def generateSettingContent(params):
    template='''batch
10
begin
1
num of epoch
59
mark point to write output
59
mode (type of output data)
0
num_train
<numtrain>
num_cv
<numcv>
num_test
<numtest>
output
<output>
parameter file
<paramFile>
fx_train  (x -training file)
<xtrain>
fx_CV     (x - validation file)
<xcv>
fx_test   (x - test file)
<xtest>
fy_train  (y - training file)
<ytrain>
fy_CV     (y- validation file)
<ycv>
fy_test   (y- test file)
<ytest>
alpha
0.1
beta
0.7
active function(ReLU, tanh)
tanh
// p1: epoch, p2: mark - export from this round, mode: 0 - export probabilities, 1- predicting results, 2-vector
'''
    for pname in params:
        key ='<'+pname+'>'
        value = str(params[pname])
        template = template.replace(key, value)
    print template

# generateSettingContent({'numtrain':10, 'numcv':20, 'numtest':30, 'output':2,
#                         'paramFile':'paramFile','xtrain': 'traindata', 'xcv':'cvdata', 'xtest':'testdata',
#                         'ytrain':'ytrain', 'ycv':'ycv','ytest': 'ytest'})