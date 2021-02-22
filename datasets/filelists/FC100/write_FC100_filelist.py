import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random

cwd = os.getcwd()
data_path = join(cwd,'images')
savedir = './'
dataset_list = ['base','val','novel']


for dataset in dataset_list:
    file_list = []
    label_list = []
    for i, classfile_name in enumerate(os.listdir(data_path)):

        if dataset in classfile_name:
            file_list.append(os.path.join(data_path, classfile_name))
            label_list.append(int(classfile_name.split('_')[1]))

    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in label_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in file_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in label_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)
