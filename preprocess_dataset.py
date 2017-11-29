import sys

from preprocess_utils import *

def run(dirname):
    remove_useless_images.run(dirname)
    convert_bmp_to_jpg.run(dirname)
    change_ext_in_xml.run(dirname, dirname)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        DIR = 'C:/Projects/Medical_image/Endoscopic/DATA_edit/detection_1127/DATA_A/test'
    else:
        DIR = sys.argv[1]
    run(DIR)
