import os
import shutil

path ='D:/StackOverFlowData/Data/'

def getLabelledSourceCode(sourcepath='', destpath='', labelfile=''):
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
    reader = open(labelfile,'r')
    for line in reader:
        items = line.split(',')
        filename = items[0]+ '.txt'
        if filename not in files.keys():
            print 'not found: ', filename
        else:
            shutil.copy2(files[filename], destpath+filename)


    reader.close()
    return

if __name__ == "__main__":
    getLabelledSourceCode(path+'SO_Codes/', destpath=path + 'LabelledData/', labelfile=path + '2017-02-20_09-51-31.290+0900.all_labels.csv')
