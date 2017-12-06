"""
convert xml files to csv format file
"""
import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd
import random

TRAIN_RATIO = 0.8

def run(csv_name, image_path):
    """
    This function get images and xmls from 'image_path/data/*' for train & validation and 'image_path/test/*' for test
    Args:
        csv_name: output csv_name
        image_path: this path should contain '/data' and '/test'
    Outputs:
        it produces 3 kinds of csv files:
            (csv_name)_train_*.csv
            (csv_name)_validation_*.csv
            (csv_name)_test.csv
    """
    column_name = ['filename', 'width', 'height', 'channels', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    
    trainval_path = os.path.join(image_path, 'data')
    test_path = os.path.join(image_path, 'test')

    gen_csv_for_trainval(csv_name, trainval_path, column_name)
    gen_csv_for_test(csv_name, test_path, column_name)

    print('Successfully converted xml to csv.')
    return


def gen_csv_for_trainval(csv_name, image_path, column_name):
    benign_list = glob.glob(image_path + '/benign/*.xml')
    cancer_list = glob.glob(image_path + '/cancer/*.xml')
    
    for i in range(5):
        benign_train, benign_validation = split_list_by_random(benign_list, TRAIN_RATIO, random_seed=100+i)
        cancer_train, cancer_validation = split_list_by_random(cancer_list, TRAIN_RATIO, random_seed=100+i)

        train_list = []
        validation_list = []

        train_list = parse_xml(benign_train + cancer_train)
        validation_list = parse_xml(benign_validation + cancer_validation)
        
        train_df = pd.DataFrame(train_list, columns=column_name)
        validation_df = pd.DataFrame(validation_list, columns=column_name)
        
        train_df.to_csv(csv_name + '_train_' + str(i) + '.csv', index=None)
        validation_df.to_csv(csv_name + '_validation_' + str(i) + '.csv', index=None)


def gen_csv_for_test(csv_name, image_path, column_name):
    test_benign_list = glob.glob(image_path + '/benign/*.xml')
    benign_test, _ = split_list_by_random(test_benign_list, 1)
    test_cancer_list = glob.glob(image_path + '/cancer/*.xml')
    cancer_test, _ = split_list_by_random(test_cancer_list, 1)

    test_list = []
    test_list = parse_xml(benign_test + cancer_test)
    test_df = pd.DataFrame(test_list, columns=column_name)
    test_df.to_csv(csv_name + '_test.csv', index=None)


def split_list_by_random(list_in, ratio, random_seed=100):
    random.seed(random_seed)
    random.shuffle(list_in)
    return split_list(list_in, ratio)


def split_list(list_in, ratio):
    split_index = round(len(list_in)*ratio)
    return (list_in[:split_index], list_in[split_index:])


def parse_xml(xml_files):
    xml_list = []
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        xml_filename = os.path.split(xml_file)[-1]
        img_filename = root.find('filename').text
        if os.path.splitext(img_filename)[0] != os.path.splitext(xml_filename)[0]:
            root.find('filename').text = os.path.splitext(xml_filename)[0] + os.path.splitext(img_filename)[-1]
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     int(root.find('size')[2].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                    )
            xml_list.append(value)
    return xml_list


if __name__ == '__main__':
    csv_name = 'C:/Projects/Medical_image/Endoscopic/DATA_edit/detection_1127/DATA_A/medical_A'
    image_path = os.path.join('C:/Projects/Medical_image/Endoscopic/DATA_edit/detection_1127/DATA_A') #folder name

    run(csv_name, image_path)
