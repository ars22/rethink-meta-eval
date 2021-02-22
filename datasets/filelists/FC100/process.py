from PIL import Image
import numpy as np
from tqdm import tqdm
import pickle

def load_data(file):
    try:
        with open(file, 'rb') as fo:
            data = pickle.load(fo)
        return data
    except:
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data

a = load_data('FC100_train.pickle')
for i, img in tqdm(enumerate(a['data']), total=len(a['data'])):
    img = Image.fromarray(img, 'RGB')
    img.save('images/base_%.3d_%.5d.png' % (a['labels'][i], i))

a = load_data('FC100_test.pickle')
for i, img in tqdm(enumerate(a['data']), total=len(a['data'])):
    img = Image.fromarray(img, 'RGB')
    img.save('images/novel_%.3d_%.5d.png' % (a['labels'][i], i))


a = load_data('FC100_val.pickle')
for i, img in tqdm(enumerate(a['data']), total=len(a['data'])):
    img = Image.fromarray(img, 'RGB')
    img.save('images/val_%.3d_%.5d.png' % (a['labels'][i], i))
