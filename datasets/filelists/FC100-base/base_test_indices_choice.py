import numpy as np
from process import load_data

unique_base_classes = set(load_data('FC100_train.pickle')['labels'])
np.random.seed(seed=42)

base_test_image_indices = {}
for cl in sorted(unique_base_classes):
    base_test_image_indices[cl] = sorted(np.random.choice(a=list(range(1, 601)), size=100, replace=False))
    # print(cl, base_test_image_indices[cl])

import pickle
with open('base_test_indices.pickle', 'wb') as file:
    pickle.dump(obj=base_test_image_indices, file=file)