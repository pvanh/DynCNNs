import random

import GraphData_IO

# datapath ='C:/Users/anhpv/Desktop/CFG/GraphJson/'
datapath ='D:/JsonAST_Graph/'
# load token types dictionary
toktypeDict = GraphData_IO.LoadTokenTypeDict(filename=datapath + 'toktypeDict.txt')
numDis = 300
numOut = 104

numView =1
numCon =[300,600]

datafiles ={}
datafiles['train_virus'] = datapath+'Training/Virus.json'
datafiles['train_nonvirus'] = datapath+'Training/NonVirus.json'
datafiles['test_virus'] = datapath+'Testing/Virus.json'
datafiles['test_nonvirus'] = datapath+'Testing/NonVirus.json'

# groups =['g_unknown','g_control','g_arithmetic','g_call','g_move','g_return','g_cond_jump','g_jump']
#
# for g in groups:
#     vec = [random.uniform(-1, 1) for v in xrange(30)]
#     vec = [str(i) for i in vec]
#     print  g +' '+' '.join(vec)
# numTok = 1424
# print '@relation dataset\n'
# for i in range(numTok):
#     print '@attribute F'+str(i+1)+' numeric'
# print '@attribute F'+str(numTok+1)+'{True, False}'
# print '\n@data\n'