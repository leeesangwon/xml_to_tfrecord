"""
convert bmp files to jpg files
"""
import os
import sys
from PIL import Image

def run(dirname):
    for (dirpath, _, filenames) in os.walk(dirname):
        for filename in filenames:
            ext = os.path.splitext(filename)[-1]
            if ext == '.bmp':
                img = Image.open(os.path.join(dirpath, filename))
                newfilename = os.path.splitext(filename)[0] + '.jpg'
                img.save(os.path.join(dirpath, newfilename))
                os.remove(os.path.join(dirpath, filename))


if __name__ == '__main__':
    assert len(sys.argv) == 1 or len(sys.argv) == 2, 'ERROR: Too many arguments'
    if len(sys.argv) == 1:
        DIR = os.getcwd()
    else:
        DIR = sys.argv[1]

    run(DIR)
