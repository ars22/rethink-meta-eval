import glob
import json
import os
import numpy as np

pathname = os.getcwd()
print(pathname)


########## meta-train ##########
base = {'label_names': [] , 'image_names':[] , 'image_labels':[]}
base_test = {'label_names': [] , 'image_names':[] , 'image_labels':[]}
f = open(pathname + '/cifar100/splits/bertinetto/train.txt')
classes = f.readlines()

np.random.seed(seed=11)
count = 0
for each in classes:
    each = each.strip()
    indices_base_test = set(np.random.choice(a=600, size=100, replace=False))
    base['label_names'].append(each)
    base_test['label_names'].append(each)
    files = sorted(glob.glob( pathname + '/cifar100/data/' + each + '/*'))
    for idx, image_name in enumerate(files):
        if idx in indices_base_test:
            base_test['image_names'].append(image_name)
            base_test['image_labels'].append(count)
        else:
            base['image_names'].append(image_name)
            base['image_labels'].append(count)
    count +=1


json.dump(base , open('base.json','w'))
json.dump(base_test , open('base_test.json','w'))


########## meta-val ##########
val = {'label_names': [] , 'image_names':[] , 'image_labels':[]}
f = open(pathname + '/cifar100/splits/bertinetto/val.txt')
classes = f.readlines()

count = 64
for each in classes:
    each = each.strip()
    val['label_names'].append(each)
    files = sorted(glob.glob( pathname + '/cifar100/data/' + each + '/*'))
    for image_name in files:
        val['image_names'].append( image_name)
        val['image_labels'].append(count)
    count +=1


json.dump(val , open('val.json','w'))


########## meta-test ##########
test = {'label_names': [] , 'image_names':[] , 'image_labels':[]}
f = open(pathname + '/cifar100/splits/bertinetto/test.txt')
classes = f.readlines()

count = 80
for each in classes:
    each = each.strip()
    test['label_names'].append(each)
    files = sorted(glob.glob( pathname + '/cifar100/data/' + each + '/*'))
    for image_name in files:
        test['image_names'].append( image_name)
        test['image_labels'].append(count)
    count +=1

json.dump(test , open('novel.json','w'))
