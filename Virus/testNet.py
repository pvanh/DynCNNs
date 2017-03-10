import GraphData_IO
import main_MultiChannelGCNN as GCNN
# import main_GCNN as GCNN
import gcnn_params as params
datapath = params.datapath
word_dict, vectors, numFea = GraphData_IO.LoadVocab(vocabfile=datapath + 'tokvec.txt')

def testNet(filename):
    graph = GraphData_IO.getGraph(filename=filename)

    layers = GCNN.InitByNodes(graph = graph , word_dict = word_dict)
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
                    '(xnum= ', c.xnum, ', ynum= ', c.ynum, '), Wid = ', c.Widx[0], '(Woef=', c.Wcoef,')'
            else:
                print "        ", c.xlayer.name, " -> ", '|', \
                    '(xnum= ', c.xnum, ', ynum= ', c.ynum, ')'

if __name__ == "__main__":
    # print GCNN.Wconv_root[0][1]
    # print GCNN.Wconv_neighbor
    # print GCNN.Bconv
    filename ='C:/Users/anhpv/Desktop/CFG/test.dot'
    testNet(filename=filename)
    # arr =[[[1,2,3,4],[2,3,2]],[2,3],[5,6]]
    # print arr[0][0]