import sys

from preprocess_utils import *

def run(dirname):
    remove_useless_images.run(dirname)
    convert_bmp_to_jpg.run(dirname)
    change_ext_in_xml.run(dirname, dirname)


if __name__ == '__main__':
    DIR = sys.argv[1]
    run(DIR)
