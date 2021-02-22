import numpy as np
import os
import json
import random

cwd = os.getcwd()
data_path = os.path.join(cwd,'images')

savedir = './'
dataset_list = ['base', 'basetest','val','novel']


for dataset in dataset_list:
    file_list = []
    label_list = []
    for i, classfile_name in enumerate(os.listdir(data_path)):
        tp, cl, idx = classfile_name.split('_')
        cl = int(cl)
        if dataset == tp:
            file_list.append(os.path.join(data_path, classfile_name))
            label_list.append(cl)
    
    with open(savedir + dataset + ".json", "w") as f:
        json_object = {
            'label_names': label_list,
            'image_names': file_list,
            'image_labels': label_list
        }
        json.dump(json_object, f)
    
    print(dataset)
    print(f'image_names: {len(file_list)}\nimage_labels: {len(label_list)}')

    '''
    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    for i, item in enumerate(label_list):
        fo.write(f'"{item}"')
        if i != len(label_list) - 1: # json does not allow trailing commas
            fo.write(',')
    fo.write('],\n')

    fo.write('"image_names": [')
    # fo.writelines(['"%s",' % item  for item in file_list])
    # fo.seek(0, os.SEEK_END)
    # fo.seek(fo.tell()-1, os.SEEK_SET)
    for i, item in enumerate(file_list):
        fo.write(f'"{item}"')
        if i != len(label_list) - 1: # json does not allow trailing commas
            fo.write(',\n')
    fo.write('],\n')

    fo.write('"image_labels": [')
    for i, item in enumerate(label_list):
        fo.write(f'"{item}"')
        if i != len(label_list) - 1: # json does not allow trailing commas
            fo.write(',')
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)
    '''
