import os
import shutil

path ='D:/StackOverFlowData/Data/'

def getLabelledSourceCode(sourcepath='', destpath='', anotatedfile='', errorLabel='16'):
    # get all files in source path
    files ={}
    dircount =0
    # get files in current dir
    for onefile in os.listdir(sourcepath):
        files[onefile] = sourcepath+onefile
    #get files in sub dirs
    for subdir in os.listdir(sourcepath):
        if not subdir.endswith('/'):
            subdir = subdir + '/'
        dircount += 1
        for onefile in os.listdir(sourcepath + subdir):
            files[onefile] = sourcepath + subdir+ onefile
    print 'sub directories = ', dircount

    # copy file to the destination
    reader = open(anotatedfile,'r')
    labels = open(destpath+'0_labels','w')
    for line in reader:
        items = line.split(',') # file name, label, comment
        if items[1] == errorLabel: # ignore syntax error program
            continue
        filename = items[0]+ '.txt'
        if filename not in files.keys():
            print 'not found: ', filename
        else:
            labels.write(filename+','+ items[1]+'\n')
            shutil.copy2(files[filename], destpath+filename)


    reader.close()
    labels.close()
    return
def splitDatabyLabels(sourceDir='', desDir='', nonDefectLabels=['1']):
    # read labels
    labels = {}
    f = open(sourceDir+'0_labels')
    for line in f:
        file_label = line.strip().split(',')
        labels[file_label[0]] = file_label[1]
    f.close()

    # create destination folder
    nonDefectDir = desDir + '1/'
    if not os.path.exists(nonDefectDir):
        os.makedirs(nonDefectDir)

    defectDir = desDir + '2/'
    if not os.path.exists(defectDir):
        os.makedirs(defectDir)

    for onefile in os.listdir(sourceDir):
        if not onefile.endswith('.txt'):
            continue
        if onefile.find('_a') != -1: # the correct answer file
            shutil.copy2(sourceDir+ onefile, nonDefectDir + onefile)
            continue

        if onefile not in labels:
            print onefile
            continue
        l = labels[onefile]
        if l in nonDefectLabels:
            shutil.copy2(sourceDir + onefile, nonDefectDir + onefile)
        else:
            shutil.copy2(sourceDir + onefile, defectDir + onefile)

if __name__ == "__main__":
    #getLabelledSourceCode(path+'SO_Codes/', destpath=path + 'AnnotatedData/', anotatedfile=path + '2017-02-20_09-51-31.290+0900.all_labels.csv')
    splitDatabyLabels(sourceDir=path+'AnnotatedData/', desDir=path+'Experiments/data/')
    print 'Done'