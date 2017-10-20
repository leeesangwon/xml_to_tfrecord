"""
change ext information of image in xml file from bmp to jpg
""" 
import os
import re
import sys
import xml.etree.ElementTree as ET

# Change extension
FROM = 'bmp'
TO = 'jpg'

def run(dir_in, dir_out):
    pathlist_in = list()
    pathlist_in = search_xml(dir_in)

    for (path_in, name_in) in pathlist_in:
        xml_in = ET.parse(path_in+"/"+name_in)
        xml_out = change_ext(xml_in, FROM, TO)
        path_out = dir_out+"/"+path_in[len(dir_in)+1:]
        if not os.path.isdir(path_out):
            os.makedirs(path_out)
        xml_out.write(path_out + "/" + name_in)


def search_xml(dirname):
    '''
    This function finds xml files within input directory.
    '''
    pathlist = list()

    for (path, _, files) in os.walk(dirname):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.xml':
                path = path.replace("\\", "/")
                pathlist.append((path, filename))

    return pathlist


def change_ext(xml_in, from_, to_):
    xml_out = xml_in
    filename_tag = xml_out.find(".//filename")
    path_tag = xml_out.find(".//path")
    filename_tag.text = re.sub(r"\."+from_, "."+to_, filename_tag.text)
    path_tag.text = re.sub(r"\."+from_, "."+to_, path_tag.text)
    return xml_out

if __name__ == '__main__':

    assert len(sys.argv) == 3, \
    "USAGE: python xml_merge.py <output dir> <input dir>"

    XML_DIR_OUT = sys.argv[1]
    XML_DIR_IN = sys.argv[2]

    run(XML_DIR_IN, XML_DIR_OUT)
