from PIL import Image
import os
from tqdm import tqdm
import pickle
import collections

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

if __name__ == '__main__':
    data_object_to_name = {
        'FC100_train.pickle': 'base',
        'FC100_val.pickle': 'val',
        'FC100_test.pickle': 'novel',
    }

    if not os.path.exists('./images'):
        os.makedirs('./images')

    import pickle
    with open('base_test_indices.pickle', 'rb') as file:
        base_test_indices = pickle.load(file)

    for data_object, name in data_object_to_name.items():
        a = load_data(data_object)
        class_counts = collections.Counter()
        for i, img in tqdm(enumerate(a['data']), total=len(a['data'])):
            cl = a['labels'][i]
            img = Image.fromarray(img, 'RGB')
            class_counts[cl] += 1
            if name == 'base' and class_counts[cl] in base_test_indices[cl]:
                # if the index within the class cl is in the base_test_indices of the class cl
                # put it in basetest, otherwise put in base.
                img.save(f'images/basetest_{cl:03d}_{class_counts[cl]:03d}.png')
            else:
                img.save(f'images/{name}_{cl:03d}_{class_counts[cl]:03d}.png')