"""
convert xml files to csv format file
"""
import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd
import random

TRAIN_RATIO = 1.0

def run(image_path, csv_name, num_of_cross_val=5):
    """
    This function get images and xmls from 'image_path/data/*' for train & validation and 'image_path/test/*' for test
    Args:
        image_path: this path should contain '/data' and '/test'
        csv_name: output csv_name
    Outputs:
        it produces 3 kinds of csv files:
            (image_path)/(csv_name)_train_*.csv
            (image_path)/(csv_name)_validation_*.csv
            (image_path)/(csv_name)_test.csv
    """
    image_path = os.path.abspath(image_path)
    column_name = ['filename', 'width', 'height', 'channels', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    
    trainval_path = os.path.join(image_path, 'data')
    test_path = os.path.join(image_path, 'test')
    csv_path = os.path.join(image_path, csv_name)

    _gen_csv_for_trainval(csv_path, trainval_path, column_name, num_of_cross_val, TRAIN_RATIO)
    _gen_csv_for_test(csv_path, test_path, column_name)

    print('Successfully converted xml to csv.')
    return


def _gen_csv_for_trainval(csv_path, image_path, column_name, num_of_cross_val, train_ratio):
    benign_list = glob.glob(os.path.join(image_path, 'benign', '*.xml'))
    cancer_list = glob.glob(os.path.join(image_path, 'cancer', '*.xml'))
    csv_path_pattern = csv_path + '_%s_%d.csv'

    for i in range(num_of_cross_val):
        benign_train, benign_validation = _split_list_by_random(benign_list, train_ratio, random_seed=100+i)
        cancer_train, cancer_validation = _split_list_by_random(cancer_list, train_ratio, random_seed=100+i)

        train = benign_train + cancer_train
        validation = benign_validation + cancer_validation
        
        random.seed(100+i)
        random.shuffle(train)
        random.shuffle(validation)

        train_list = []
        validation_list = []

        train_list = _parse_xml(train)
        validation_list = _parse_xml(validation)
        
        train_df = pd.DataFrame(train_list, columns=column_name)
        validation_df = pd.DataFrame(validation_list, columns=column_name)
        
        train_df.to_csv(csv_path_pattern % ('train', i), index=None)
        validation_df.to_csv(csv_path_pattern % ('validation', i), index=None)


def _gen_csv_for_test(csv_path, image_path, column_name):
    test_benign_list = glob.glob(os.path.join(image_path, 'benign', '*.xml'))
    benign_test, _ = _split_list_by_random(test_benign_list, 1)
    test_cancer_list = glob.glob(os.path.join(image_path, 'cancer', '*.xml'))
    cancer_test, _ = _split_list_by_random(test_cancer_list, 1)

    test_list = []
    test_list = _parse_xml(benign_test + cancer_test)
    test_df = pd.DataFrame(test_list, columns=column_name)
    test_df.to_csv(csv_path + '_test.csv', index=None)


def _split_list_by_random(list_in, ratio, random_seed=100):
    random.seed(random_seed)
    random.shuffle(list_in)

    def _split_list(list_in, ratio):
        split_index = round(len(list_in)*ratio)
        return (list_in[:split_index], list_in[split_index:])

    return _split_list(list_in, ratio)


def _parse_xml(xml_files):
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
    image_path = 'C:/Projects/Medical_image/Endoscopic/DATA_edit/detection_1127/DATA_A' #folder name
    csv_name = 'medical_A'
    num_of_cross_val = 5
    
    run(image_path, csv_name, num_of_cross_val)
