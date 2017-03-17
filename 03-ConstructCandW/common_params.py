import cPickle as p
# settings
import os

ignoreDecl = False # prune declaration branches
reConstruct = False # rename While, DoWhile, For ==> Loop
tokenMapFile = '../tokenMap.txt'
tokenMap = p.load(open(tokenMapFile))

datadir =  'D:/data/original_data/' # source code directory
jsondir = 'C:/Users/anhpv/Desktop/data/' #jsondir = 'C:/Users/anhpv/Desktop/data/' #
xypath = jsondir + 'xy/'

dataname ='OJ'

jsonfold={}
jsonfold['train'] = '_train_AST.json'
jsonfold['CV'] = '_CV_AST.json'
jsonfold['test'] = '_test_AST.json'
