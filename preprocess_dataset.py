"""
Preprocess data.
For trainval:
    1. Remove image files which don't have matched xml files.
    2. Convert bmp files to jpg files.
    3. Change ext information of image in xml file from bmp to jpg.
For test:
    Only convert bmp files to jpg files.
Args:
    dirname:
        Directory name which include data.
    is_test_data:
        Type of the input data.
        If is_test_data is True, ONLY DO convert bmp files to jpg files.
"""
import sys

from preprocess_utils import *

def run(dirname, is_test_data=False):
    if is_test_data:
        convert_bmp_to_jpg.run(dirname)
        return

    remove_useless_images.run(dirname)
    convert_bmp_to_jpg.run(dirname)
    change_ext_in_xml.run(dirname, dirname)
    return


if __name__ == '__main__':
    if len(sys.argv) < 2:
        DIR = 'C:/Projects/Medical_image/Endoscopic/DATA_edit/detection_1127/DATA_A/test'
    else:
        DIR = sys.argv[1]
    run(DIR, is_test_data=True)
