import sys

import re


class GVertex:
    def __init__(self, id=None, name='', token='', toktype='ASM', content=''):
        self.id = id
        self.name = name
        self.token = token
        self.toktype = toktype
        self.content = content
    def getViews(self, toktypeDict):
        data =[self.token]
        if self.token in toktypeDict:
            data.extend(toktypeDict[self.token])
        else:
            data.append('g_unknown')
        return data
    @staticmethod
    def fromContent(vinfor):
        idx2 = len(vinfor)
        if '_' in vinfor:
            idx2 = vinfor.index('_')
        idx1 = 0
        toktype='API'
        if vinfor.startswith('a0x'):
            toktype='ASM'
            idx1 = 11
            if re.match('a0x(\d|[a-e]|f){12,}', vinfor, flags=0):
                idx1 = 19

        token = vinfor[idx1:idx2]
        vertex = GVertex(id=-1, name=vinfor, token=token, toktype=toktype, content='')
        return vertex
    def show(self, buf = sys.stdout):
        buf.write(str(self.id) +'-'+ self.token+' ('+self.name+'-'+ self.content+')')
    def dump(self):
        return {'id':self.id,'name':self.name, 'token':self.token,'toktype':self.toktype, 'content':self.content}
class Graph:
    Vs =None
    Es =None
    def __init__(self, vs=None, es=None, label = -1):
        if vs is None:
            self.Vs ={}
        else:
            self.Vs = vs
        if es is None:
            self.Es =[]
        else:
            self.Es = es
        self.label = label
    def addVetex(self, v=None):
        self.Vs[v.id] = v
    def addEdge(self, node1, node2):
        self.Es.append((node1.id, node2.id))
    def addEdgebyID(self, node1_id, node2_id):
        self.Es.append((node1_id, node2_id))
    def show(self, buf = sys.stdout):
        buf.write('Vertexes\n')
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

    def dump(self):
        vertexes=[]
        for v in self.Vs.values():
            vertexes.append(v.dump())
        edges =[]
        for (v1,v2) in self.Es:
            edges.append((v1, v2))
        return {'V':vertexes,'E':edges,'label':self.label}

    @staticmethod
    def load(dumped_obj):
        g = Graph()
        # get vertexes
        for v in dumped_obj['V']:
            g.addVetex(GVertex(id=v['id'], name=v['name'], token=v['token'], toktype=v['toktype'], content=v['content']))
        for v1_id, v2_id in dumped_obj['E']:
            g.addEdgebyID(v1_id, v2_id)
        g.label = dumped_obj['label']
        return g
    # @staticmethod
    # def loadGraph(file=''):
    #     nodename_dict ={}
    #     g = Graph()
    #
    #     return g