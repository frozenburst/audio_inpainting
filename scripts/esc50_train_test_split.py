#!/usr/bin/env python
# coding: utf-8
from tqdm import tqdm
from shutil import copyfile

import os.path as op
import os
import glob
import random
import argparse

# 1/8 to make sure balanced data.
_NUM_TEST = 0.125
_NUM_CLASS = 50
_NUM_PART = 5
parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='/your/workspace/datasets/ESC-50-master/audio', type=str,
                    help='The data root')
parser.add_argument('--train_path', default='/your/workspace/datasets/ESC-50-master/split/train', type=str,
                    help='The place for training data')
parser.add_argument('--test_path', default='/your/workspace/datasets/ESC-50-master/split/test', type=str,
                    help='The place for testing data.')


if __name__ == "__main__":
    args = parser.parse_args()
    folder_dir = args.folder_path
    train_path = args.train_path
    test_path = args.test_path

    if op.isdir(folder_dir) is False:
        raise ValueError("The data folder is not exist with path:", folder_dir)
    # mask dir of split
    if op.isdir(op.dirname(train_path)) is False:
        os.mkdir(op.dirname(train_path))
    if op.isdir(train_path) is False:
        os.mkdir(train_path)
    if op.isdir(test_path) is False:
        os.mkdir(test_path)

    audio_filenames = glob.glob(f'{folder_dir}/*.wav')
    num_files = len(audio_filenames)
    print(f'size of esc50 is {num_files}')

    # [1, 2, 3, ...]
    part_ID = [i+1 for i in range(_NUM_PART)]
    # [0, 1, 2, ...]
    category_ID = [i for i in range(_NUM_CLASS)]

    num_of_parts = 8
    test_split_nun = int(_NUM_TEST * num_of_parts)

    # Due to the lack of data: (2000 * 5sec)
    # I split the train test dataset with average in source.
    # In general, the fragments from same source should not put in
    # both train and test set for generalization (refer to Esc-50),
    # however, consider that would be
    # hard for GAN to generate the whole complete sound without seeing
    # certain kinds of data in that recording environments.

    # If the dataset is like a music record with large amount,
    # this should be split in just normal random.
    random.seed(1373)
    for pid in tqdm(part_ID):
        for cid in tqdm(category_ID):
            part_of_files = glob.glob(f'{folder_dir}/{pid}-*-{cid}.wav')
            # 50 semantical classes (with 40 examples per class)
            # loosely arranged into 5 major categories
            if len(part_of_files) != num_of_parts:
                raise ValueError("Files in parts as not expected:", part_of_files)
            random.shuffle(part_of_files)
            train_files = part_of_files[test_split_nun:]
            test_files = part_of_files[:test_split_nun]

            for f in train_files:
                dst = op.join(train_path, op.basename(f))
                copyfile(f, dst)
            for f in test_files:
                dst = op.join(test_path, op.basename(f))
                copyfile(f, dst)
