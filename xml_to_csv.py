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
    benign_list = glob.glob(image_path + '/benign/*.xml')
    benign_train, benign_validation = split_list_by_random(benign_list, TRAIN_RATIO)
    
    cancer_list = glob.glob(image_path + '/cancer/*.xml')
    cancer_train, cancer_validation = split_list_by_random(cancer_list, TRAIN_RATIO)
    
    train_list = []
    validation_list = []

    train_list = parse_xml(benign_train + cancer_train)
    validation_list = parse_xml(benign_validation + cancer_validation)
    
    column_name = ['filename', 'width', 'height', 'channels', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    
    train_df = pd.DataFrame(train_list, columns=column_name)
    validation_df = pd.DataFrame(validation_list, columns=column_name)
    
    train_df.to_csv(csv_name + '_train_1.csv', index=None)
    validation_df.to_csv(csv_name + '_validation_1.csv', index=None)
    
    print('Successfully converted xml to csv.')
    return


def split_list_by_random(list_in, ratio):
    random.seed(100)
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
    image_path = os.path.join('C:/Projects/Medical_image/Endoscopic/DATA_edit/detection_1127/DATA_A/data') #folder name

    run(csv_name, image_path)
