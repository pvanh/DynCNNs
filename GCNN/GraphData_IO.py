import json
import random
import struct
from operator import itemgetter

import numpy as np
import os

import re

import commonFunctions
from Graph import Graph, GVertex

def LoadTokenTypeDict(filename =''):
    file = open(filename, "r")
    idx =0
    dict={}
    file.readline() # ignore header
    for line in file:
        items = line.rstrip().split()
        if len(items)>=2:
            dict[items[0]]= items[1:]
    return dict

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
        if len(items[1:]) != vecsize:
            word=''
            vectors.append(items)
        else:
            word= items[0]
            vectors.append(items[1:])
        dict[word] = idx
        idx = idx +1

    vectors = np.reshape(vectors,-1)
    # convert to float
    vectors = [float(i) for i in vectors]
    return dict, vectors, vecsize
def getGraphFromTextFile(filename =''):
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
        if '[' not in line:
            break
        idx = line.index('[')
        line = line[:idx]

        edge = line.split(' -> ')

        if len(edge) ==1: # vertex
            vertex = GVertex.fromContent(edge[0])
            vertex.id = nodeid
            # add to dictionary
            nodename_dict[vertex.name] = vertex
            #add to graph
            g.addVetex(vertex)

            nodeid +=1
        else:
            name1 = edge[0].strip()
            name2 = edge[1].strip()
            # add to graph
            if name1 not in nodename_dict.keys():
                vertex = GVertex.fromContent(name1)
                vertex.id = nodeid
                # add to dictionary
                nodename_dict[vertex.name] = vertex
                # add to graph
                g.addVetex(vertex)
                nodeid+=1
                print 'Not found vertex: ', name1
            if name2 not in nodename_dict.keys():
                vertex = GVertex.fromContent(name2)
                vertex.id = nodeid
                # add to dictionary
                nodename_dict[vertex.name] = vertex
                # add to graph
                g.addVetex(vertex)
                nodeid += 1
                print 'Not found vertex: ', name2
            g.addEdge(nodename_dict[name1], nodename_dict[name2])

    return g
def dataStatistic(datadir=''):

    vertex_cout =[]
    edge_count =[]
    tokens ={}
    for subdir in os.listdir(datadir):
        if not subdir.endswith('/'):
            subdir = subdir + '/'

        if os.path.isfile(datadir + subdir[:len(subdir) - 1]):
            continue


        for onefile in os.listdir(datadir + subdir):
            filename = onefile
            onefile = datadir + subdir + onefile

            if filename.endswith(".dot"):
                g = getGraphFromTextFile(onefile)
                vertexes = g.getVertexes().values()
                for v in vertexes:
                    tokens[v.token] = v.token
                vertex_cout.append(len(g.getVertexes()))
                edge_count.append(len(g.getEdges()))
    # write statistics
    out = open(datadir+'/statistic.csv','w')
    out.write('vertex, edge\n');
    for idx in xrange(len(vertex_cout)):
        out.write(str(vertex_cout[idx])+','+str(edge_count[idx])+'\n')
    out.close()
    #write tokens
    out = open(datadir + '/token', 'w')
    for tok in tokens:
        out.write(tok+'\n');
    out.close()
def writeGraph2Json(datadir, out):
    jsonObjs =[]
    for onefile in os.listdir(datadir):
        if onefile.endswith(".dot"):
            g = getGraphFromTextFile(datadir + onefile)
            jsonObjs.append(g.dump())

     #write to file
    with open(out, 'w') as outfile:
        json.dump(jsonObjs, outfile)
def readGraphFromJson(jsonFile=''): # 1: non-virus, 2: virus
    graphs=[]
    with open(jsonFile, 'r') as f:
        jsonObjs = json.load(f)
        for obj in jsonObjs:
            g = Graph.load(obj)
            graphs.append(g)
    return graphs

def createTokenVecs(datafiles='', out='', vecsize=30):
    tokdict ={}
    for file in datafiles:
        with open(file, 'r') as f:
            jsonObjs = json.load(f)
            for obj in jsonObjs:
                g = Graph.load(obj)
                for v in g.Vs.values():
                    tokdict[v.token] = v.toktype
    # write token to file
    tokens =[]
    for tok in tokdict:
        tokens.append((tok, tokdict[tok]))
    # sort by type: ASM or API
        tokens=sorted(tokens, key=itemgetter(1))

    f = open(out, 'w')
    f.write(str(len(tokens))+' '+ str(vecsize)+'\n')
    for tok, toktype in tokens:
        f.write(tok + ' ')
        vec = [random.uniform(-1, 1) for v in xrange(vecsize)]
        vec = [str(i) for i in vec]
        f.write(' '.join(vec))
        f.write('\n')
    f.close()
def searchContentInFile(datadir='', searchValue=''):
    for onefile in os.listdir(datadir):
        if onefile.endswith('.dot'):
            with open(datadir+onefile,'r') as f:
                content = f.read()
                if content.find(searchValue) != -1:
                    print datadir+onefile
                    #print content
                    #break
def getGraphsFromDataDir(datadir, classlabel =0):
    jsonObjs =[]
    count = 1
    for onefile in os.listdir(datadir):
        if onefile.endswith(".dot"):
            g = getGraphFromTextFile(datadir + onefile)
            if classlabel>=0:
                g.label = classlabel
            jsonObjs.append(g.dump())

            count +=1
    return jsonObjs

if __name__ == "__main__":

    path ='C:/Users/anhpv/Desktop/CFG/'
    # fold ='Training/'
    # name= 'GCNN'
    # datapath = path+fold

    # datapath='C:/Users/anhpv/Desktop/CFG/DataForCheckImp/'
    # writeGraph2Json(datapath+name+'/', datapath+name+'.json')

    # write token vectors
    # datapath = 'C:/Users/anhpv/Desktop/CFG/'
    # createTokenVecs(datafiles=[datapath+'NonVirus.json'],out=datapath+'tokvec.txt')

    # check content
    # searchContentInFile(path+fold+name+'/','db0c47d0addb')


    #path ='C:/Users/anhpv/Desktop/CFG_Virus/test_data/'
    #dataStatistic(path)
    #print 'Done'
    # g = getGraph(path+'faultrep.dll_model.dot')
    # g.show()
    #
    # print 'Graph 2'
    # g = getGraph(path+'test.dot')
    # g.show()
    # json_object = g.dump()
    # with open(path+'test.json', 'w') as outfile:
    #     #json.dumps([o.dump() for o in my_list_of_ipport])
    #     json.dump([json_object], outfile)
    # # print json_object
    # with open(path + 'test.json', 'r') as outfile:
    #     json_object = json.load(outfile)
    #     g= Graph.load(json_object[0])
    #     g.show()
    # path = 'C:/Users/anhpv/Desktop/CFG/'
    # g = getGraph(path + 'check1.dot')
