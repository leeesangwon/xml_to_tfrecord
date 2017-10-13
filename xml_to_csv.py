import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd


def xml_to_csv():
    csv_name = 'C:/Projects/Medical_image/Endoscopic/DATA_edit/data1013/medical_train_2.csv'
    image_path = os.path.join('C:/Projects/Medical_image/Endoscopic/DATA_edit/data1013/train') #folder name

    xml_list = []
    for xml_file in glob.glob(image_path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
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
    column_name = ['filename', 'width', 'height', 'channels', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    xml_df.to_csv(csv_name, index=None)
    print('Successfully converted xml to csv.')


if __name__ == '__main__':
    xml_to_csv()
