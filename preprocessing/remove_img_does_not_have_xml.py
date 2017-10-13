import os

def remove_img_which_does_not_have_xml(dirname):

    for (dirpath, dirnames, filenames) in os.walk(dirname):
        xmllist = []
        imglist = []
        for filename in filenames:
            ext = os.path.splitext(filename)[-1]
            if ext == '.xml':
                xmllist.append(os.path.splitext(filename)[0])
            elif ext == '.jpg' or ext == '.bmp':
                imglist.append(filename)

        for imgname in imglist:
            if not os.path.splitext(imgname)[0] in xmllist:
                os.remove(os.path.join(dirpath, imgname))


if __name__ == '__main__':
    DIR = os.getcwd()
    remove_img_which_does_not_have_xml(DIR)
