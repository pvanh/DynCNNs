import os
from shutil import copyfile
import numpy as np

def insertText(filename='', libs=''):

    file = open(filename, 'r+b')
    #return top of file
    content = file.read()
    file.seek(0)
    file.truncate()
    file.write(libs)
    file.write(content)
    file.close()

def replaceText(filename,oldString, newString):
    file = open(filename, 'r+b')
    # return top of file
    content = file.read()
    content=content.replace(oldString, newString)
    file.seek(0)
    file.truncate()

    file.write(content)
    file.close()

def addLibs():
    datadir = 'D:/data/data_Ctype/'
    # datadir ='D:/C_SourceCode/'
    #
    #     libs ="""#include<stdio.h>
    # #include<iostream>
    # #include <iomanip>
    # #include <math.h>
    # # include<stdlib.h>
    # # include<string.h>
    # using namespace std;
    # """

    count = 0
    for subdir in os.listdir(datadir):
        if not subdir.endswith('/'):
            subdir = subdir + '/'

        if os.path.isfile(datadir + subdir[:len(subdir) - 1]):
            continue

        print '!!!!!!!!!!!!!!!!!!  procount = ', subdir
        for onefile in os.listdir(datadir + subdir):
            filename = onefile
            onefile = datadir + subdir + onefile

            if filename.endswith(".c"):
                if not os.path.isfile(onefile + '.exe'):  # if not be compiled
                    count += 1
                    # addLib2FileC(filename=onefile,libs = libs)
                    # replaceText(filename= onefile,oldString='void main',newString= 'int main')
                    print filename
    print 'No. of files:', count
def copyExe(sourcedir, desdir):
    for subdir in os.listdir(sourcedir):
        if not subdir.endswith('/'):
            subdir = subdir + '/'
        subdirName = subdir.replace('/','')

        if os.path.isfile(sourcedir + subdir[:len(subdir) - 1]):
            continue

        print '!!!!!!!!!!!!!!!!!!  procount = ', subdir
        for onefile in os.listdir(sourcedir + subdir):
            filename = onefile
            onefile = sourcedir + subdir + onefile

            if filename.endswith(".exe"):
                 copyfile(onefile, desDir + subdirName+'_' + filename)
                 #print subdirName+'_' + filename
def checkFile(source1 ='', source2 ='',destdir =''  ):
    files1={}
    files2 ={}
    #delete overlap files
    # for filename in os.listdir(source1):
    #     if filename.endswith('.dot'):
    #         idx = filename.find('_')
    #         ext = filename[idx+1:]
    #         name = filename[:idx ]
    #         if ext=='test_model.dot':
    #             if os.path.exists(source1+ name+'_model.dot'):
    #                 os.remove(source1+filename)
    #                 print name
    #
    # for filename in os.listdir(source2):
    #     if filename.endswith('.dot'):
    #         idx = filename.find('_')
    #         ext = filename[idx + 1:]
    #         name = filename[:idx]
    #         if ext == 'test_model.dot':
    #             if os.path.exists(source2 + name + '_model.dot'):
    #                 os.remove(source2 + filename)
    #                 print name


    for filename in os.listdir(source1):
        if filename.endswith('.dot'):
            idx = filename.find('.')
            name = filename[:idx]
            files1[name] = source1+ filename
    # for name in files1:
    #     print name,': ' ,files1[name]


    for filename in os.listdir(source2):
        if filename.endswith('.dot'):
            idx = filename.find('.')
            name = filename[:idx]
            if name in files1:
                size1 = os.path.getsize(files1[name])
                size2 = os.path.getsize(source2+filename)
                if size1>size2:
                    copyfile(files1[name], destdir+ name+'.dot')
                else:
                    copyfile(source2+filename, destdir + name+'.dot')
            else:
                copyfile(source2 + filename, destdir + name+'.dot')
def splitTrainTest(sourcePath='',trainPath='', testPath ='' ):
    files=[]
    for subdir in os.listdir(sourcePath):
        if not subdir.endswith('/'):
            subdir = subdir + '/'

        for onefile in os.listdir(sourcePath + subdir):
            files.append((subdir, onefile,sourcePath+subdir+onefile))

    np.random.seed(314159)
    np.random.shuffle(files)

    numTrain = (int)(len(files)*0.8)
    # copy training
    for idx in range(0, numTrain):
        (subdir, name, fullname) = files[idx]
        copyfile(fullname, trainPath + subdir + name)

    # copy to test
    for idx in range(numTrain, len(files)):
        (subdir, name, fullname) = files[idx]
        copyfile(fullname, testPath + subdir + name)
if __name__ == "__main__":
    sourceDir='D:/data/data_Ctype/'
    desDir ='D:/data/data_exe/'
    #
    # source2 = 'C:/Users/anhpv/Desktop/CFG/Experiment/NonVirus/'
    # source1 ='C:/Users/anhpv/Desktop/CFG/non_virus/'
    # destdir ='C:/Users/anhpv/Desktop/CFG/Experiment/NonVirus_Merged/'
    # checkFile(source1=source1, source2= source2, destdir= destdir)

    # copyExe(sourcedir= sourceDir, desdir= desDir)
    sourcePath = 'C:/Users/anhpv/Desktop/CFG/Experiment/OriginalData/'
    trainPath = 'C:/Users/anhpv/Desktop/CFG/Experiment/Training/'
    testPath = 'C:/Users/anhpv/Desktop/CFG/Experiment/Testing/'
    splitTrainTest(sourcePath= sourcePath, trainPath=trainPath, testPath=testPath)