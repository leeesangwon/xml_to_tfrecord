''' 
remove image files which don't have matched xml files 
'''
import os
import sys

def run(dirname):
    for (dirpath, dirnames, filenames) in os.walk(dirname):
        xmllist = []
        imglist = []
        for filename in filenames:
            ext = os.path.splitext(filename)[-1]
            if ext == '.xml':
                xmllist.append(os.path.splitext(filename)[0])
            elif ext == '.jpg' or ext == '.bmp':
                imglist.append(filename)

        for imgname in imglist:
            if not os.path.splitext(imgname)[0] in xmllist:
                os.remove(os.path.join(dirpath, imgname))


if __name__ == '__main__':
    assert len(sys.argv) == 1 or len(sys.argv) == 2, 'ERROR: Too many arguments'
    if len(sys.argv) == 1:
        DIR = os.getcwd()
    else:
        DIR = sys.argv[1]

    run(DIR)
