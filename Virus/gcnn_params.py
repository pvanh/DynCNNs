import random

import Data_IO

datapath ='C:/Users/anhpv/Desktop/CFG/'
# load token types dictionary
toktypeDict = Data_IO.LoadTokenTypeDict(filename=datapath+'toktypeDict.txt')
numDis = 5
numOut = 2

numView =2
numCon =[2,3,4]

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