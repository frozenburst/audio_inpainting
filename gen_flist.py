from pathlib import Path
from tqdm import tqdm

import numpy as np
import os.path as op
import os


# Parameters
validation_split = 0.1
training_file = './data/train_list.txt'
testing_file = './data/test_list.txt'
data_pth = '/tmp3/paulolbear/dataset/ESC-50-master/spectrogram_15seg'


### Audio preprocess
train_name = []
train_specs = []
train_labels = []

test_name = []
test_specs = []
test_labels = []

n_data = len(os.listdir(data_pth))
print(f'There are {n_data} files.')

for i, filename in tqdm(enumerate(Path(data_pth).glob('*.npy'))):
    data_name = str(filename)
    data = np.load(filename)
    
    w, h, _ = data.shape
    data = data.reshape(w, h)
    
    class_name = int(op.basename(filename).split('-')[-1].split('_')[0])
    
    if i < n_data * validation_split:
        test_name.append(data_name)
        test_specs.append(data)
        test_labels.append(class_name)
    else:
        train_name.append(data_name)
        train_specs.append(data)
        train_labels.append(class_name)
train_specs = np.array(train_specs)
test_specs = np.array(test_specs)


# save as txt file
length = len(train_name)
with open(training_file, 'w') as f:
    for i in range(length):
        f.write(str(train_name[i]) + ' ' + str(train_labels[i]) +'\n')
        
length = len(test_name)
with open(testing_file, 'w') as f:
    for i in range(length):
        f.write(str(test_name[i]) + ' ' + str(test_labels[i]) +'\n')
