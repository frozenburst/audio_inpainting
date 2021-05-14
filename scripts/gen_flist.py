#!/usr/bin/env python
# coding: utf-8
from tqdm import tqdm

import os
import os.path as op
import argparse
import glob


parser = argparse.ArgumentParser()
parser.add_argument('--train_path', default='/work/r08922a13/datasets/maestro-v3.0.0/sr41k/train/preprocess', type=str,
                    help='The place for training data')
parser.add_argument('--test_path', default='/work/r08922a13/datasets/maestro-v3.0.0/sr41k/test/preprocess', type=str,
                    help='The place for testing data.')
parser.add_argument('--train_file', default='../data/maestro/train_list.txt', type=str,
                    help='The generation name of training file list.')
parser.add_argument('--test_file', default='../data/maestro/test_list.txt', type=str,
                    help='The generation name of testing file list.')


class hp:
    wav_suffix = "-wave.npy"
    mag_suffix = "-mag-raw-feats.npy"
    mel_suffix = "-mel-raw-feats.npy"
    wav_ext = ".wav"
    img_ext = ".png"


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    train_path = args.train_path
    test_path = args.test_path
    train_file = args.train_file
    test_file = args.test_file

    if op.isdir(train_path) is False:
        raise ValueError("The data folder is not exist with path:", train_path)
    if op.isdir(test_path) is False:
        raise ValueError("The data folder is not exist with path:", test_path)

    if op.isdir(op.dirname(train_file)) is False:
        os.mkdir(op.dirname(train_file))
    if op.isdir(op.dirname(test_file)) is False:
        os.mkdir(op.dirname(test_file))

    train_fnames = glob.glob(f"{train_path}/*{hp.mag_suffix}")
    test_fnames = glob.glob(f"{test_path}/*{hp.mag_suffix}")

    length = len(train_fnames)
    length_t = len(test_fnames)
    print(length, length_t)
    with open(train_file, 'w') as f:
        for i in tqdm(range(length)):
            f.write(str(train_fnames[i]) + '\n')

    with open(test_file, 'w') as f:
        for i in tqdm(range(length_t)):
            f.write(str(test_fnames[i]) + '\n')
