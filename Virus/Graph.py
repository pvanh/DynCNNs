import sys


class GNode:
    id = None
    name = None
    value = None
    content =None

    def __init__(self, id=None, name='', value='', content=''):
        self.id = id
        self.name = name
        self.value = value
        self.content = content
    def show(self, buf = sys.stdout):
        buf.write(str(self.id) +'-'+ self.value+' ('+self.name+'-'+ self.content+')')
class Graph:
    Vs =None
    Es =None

    def __init__(self, vs=None, es=None):
        if vs is None:
            self.Vs ={}
        else:
            self.Vs = vs
        if es is None:
            self.Es =[]
        else:
            self.Es = es
    def addNode(self, node=None):
        self.Vs[node.id] = node
    def addEdge(self, node1, node2):
        self.Es.append((node1.id, node2.id))
    def show(self, buf = sys.stdout):
        buf.write('Nodes\n')
        for id in self.Vs:
            self.Vs[id].show(buf= buf)
            buf.write('\n')
        buf.write('Edges\n')
        for (v1, v2) in self.Es:
            buf.write(str(v1)+' --> '+ str(v2)+'\n')
    def getVertexes(self):
        return self.Vs
    def getEdges(self):
        return self.Es
    # @staticmethod
    # def loadGraph(file=''):
    #     nodename_dict ={}
    #     g = Graph()
    #
    #     return g