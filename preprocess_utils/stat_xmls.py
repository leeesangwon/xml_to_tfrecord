"""
analyze xml files
"""
import os
import sys
import xml.etree.ElementTree as ET

def run(dire, print_flag):
    '''
    This function stat xml files
    '''
    pathlist = search_xml(dire)
    all_num_benign = 0
    all_num_malignant = 0
    num_benign_file = 0
    num_malignant_file = 0
    for path, name in pathlist:
        num_benign, num_malignant = parse_xml(path, name)
        if print_flag:
            print("\t\t%s\tbenign: %d, malignant: %d" % (name, num_benign, num_malignant))

        if num_benign != 0 and num_malignant == 0:
            num_benign_file += 1
        elif num_benign == 0 and num_malignant != 0:
            num_malignant_file += 1
                   
        all_num_benign += num_benign
        all_num_malignant += num_malignant

    if print_flag:
        print('Number of Files          benign: %d, malignant: %d, total: %d' % (num_benign_file, num_malignant_file, num_benign_file + num_malignant_file))
        print('All Number of Objects    benign: %d, malignant: %d, total: %d' % (all_num_benign, all_num_malignant, all_num_benign + all_num_malignant))
    
    return (num_benign_file, num_malignant_file, all_num_benign, all_num_malignant)


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


def parse_xml(path, name):
    xml = ET.parse(path+"/"+name)
    num_benign = 0
    num_malignant = 0
    for obj in xml.findall(".//object"):
        if obj.find("name").text == "Benign":
            num_benign += 1
        elif obj.find("name").text == "Malignant":
            num_malignant += 1

    return (num_benign, num_malignant)


def to_bool(str_in):
    """
    convert string to boolean, on: True, off: False
    """
    true_list = ['on', 'ON', 'On', 'True', 'true']
    false_list = ['off', 'OFF', 'Off', 'False', 'false']
    assert str_in in true_list + false_list, "Invalid input"
    if str_in in true_list:
        return True
    elif str_in in false_list:
        return False


if __name__ == '__main__':
    DIR_OF_XMLS = sys.argv[1]
    FLAG_PRINT = sys.argv[2] # on / off
    BOOL_PRINT = to_bool(FLAG_PRINT)
    run(DIR_OF_XMLS, BOOL_PRINT)
