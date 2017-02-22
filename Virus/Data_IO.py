import numpy as np

from Graph import GNode, Graph

def LoadVocab(vocabfile =''):
    file = open(vocabfile, "r")
    idx =0
    vectors=[]
    dict={}
    vecsize =0
    for line in file:
        items = line.rstrip().split()
        if len(items)<=2:
            vecsize = int(items[1])
            continue
        word= items[0]
        dict[word] = idx
        idx = idx +1

        vectors.append(items[1:])
    vectors = np.reshape(vectors,-1)
    # convert to float
    vectors = [float(i) for i in vectors]
    return dict, vectors, vecsize
def getGraph(filename =''):
    reader = open(filename,'r')

    #ignore 3 first rows
    reader.readline()
    reader.readline()
    reader.readline()
    #
    nodename_dict={}
    nodeid =0
    g = Graph()
    for line in reader:
        line = line.strip()
        if line =='}':
            break
        idx = line.index('[')
        line = line[:idx]

        edge = line.split(' -> ')

        if len(edge) ==1:
            idx = len(edge[0])
            if '_' in edge[0]:
                idx = edge[0].index('_')
            value = edge[0][11:idx]
            node = GNode(id= nodeid,name = edge[0],value = value,content='')
            # add to dictionary
            nodename_dict[node.name] = node
            #add to graph
            g.addNode(node)

            nodeid +=1
        else:
            name1 = edge[0].strip()
            name2 = edge[1].strip()
            # add to graph
            if name1 not in nodename_dict.keys():
                print 'Not found vertex: ', name1
            if name2 not in nodename_dict.keys():
                    print 'Not found vertex: ', name2
            g.addEdge(nodename_dict[name1], nodename_dict[name2])

    return g
if __name__ == "__main__":

    path ='C:/Users/anhpv/Desktop/CFG_Virus/non_virus/'
    g = getGraph(path+'faultrep.dll_model.dot')
    g.show()

    print 'Graph 2'
    g = getGraph(path+'debug.exe_model.dot')
    g.show()