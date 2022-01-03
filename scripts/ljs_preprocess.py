#!/usr/bin/env python
# coding: utf-8
from pathlib import Path
from tqdm import tqdm
from shutil import copyfile

import tensorflow as tf
import numpy as np
import os.path as op
import matplotlib.pyplot as plt
import os
import librosa
import glob
import random
import argparse


# 1/8 to make sure balanced data.
_NUM_TEST = 0.1
_NUM_CLASS = 0
_NUM_PART = 0
parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='/your/workspace/datasets/LJSpeech-1.1/wavs', type=str,
                    help='The data root')
parser.add_argument('--train_path', default='/your/workspace/datasets/LJSpeech-1.1/train', type=str,
                    help='The place for training data')
parser.add_argument('--test_path', default='/your/workspace/datasets/LJSpeech-1.1/test', type=str,
                    help='The place for testing data.')


class hp:
    sr = 22050  # Sampling rate. LJS:22050
    n_fft = 510     # let height of spec be 256
    win_length = n_fft
    hop_length = 256    # fix due to 510/2 = 255
    n_mels = 80
    power = 2
    max_db = 100
    ref_db = 20
    least_amp = 1e-5
    min_length = 5      # spec length 256 ~= 3 sec in sr=22050
    preprocess_dir = "preprocess"
    wav_suffix = "-wave"
    mag_suffix = "-mag-raw-feats"
    mel_suffix = "-mel-raw-feats"
    wav_ext = ".wav"
    img_ext = ".png"


def toSpec_db_norm(waveform):
    stfts = tf.signal.stft(
        waveform, frame_length=hp.win_length, frame_step=hp.hop_length, fft_length=hp.n_fft, pad_end=True)
    spec = tf.math.abs(stfts)
    spec = tf.math.pow(spec, 2)
    # To decible
    db_spec = hp.ref_db * tf.experimental.numpy.log10(tf.math.maximum(hp.least_amp, spec))
    # Normalize
    db_spec_norm = tf.clip_by_value(db_spec / hp.max_db, -1, 1)
    return db_spec_norm


def spec_denorm_deDB(spec):
    # De-normalize
    de_norm_spec = spec * hp.max_db
    # To amplitude
    de_db_spec = tf.math.pow(10.0, de_norm_spec * 0.05)
    return de_db_spec


def toMel_db_norm(deDB_denorm_spec, sr):
    to_mel_m = tf.signal.linear_to_mel_weight_matrix(hp.n_mels, deDB_denorm_spec.shape[-1], sr)
    mel = tf.tensordot(deDB_denorm_spec, to_mel_m, 1)
    # To decible
    db_mel = hp.ref_db * tf.experimental.numpy.log10(tf.math.maximum(hp.least_amp, mel))
    # Normalize
    db_mel_norm = tf.clip_by_value(db_mel / hp.max_db, -1, 1)
    return db_mel_norm


if __name__ == "__main__":
    args = parser.parse_args()
    folder_dir = args.folder_path
    train_path = args.train_path
    test_path = args.test_path

    if op.isdir(folder_dir) is False:
        raise ValueError("The data folder is not exist with path:", folder_dir)
    if op.isdir(train_path) is False:
        os.mkdir(train_path)
    if op.isdir(test_path) is False:
        os.mkdir(test_path)

    process_dir = hp.preprocess_dir
    process_train_path = op.join(train_path, process_dir)
    process_test_path = op.join(test_path, process_dir)
    if op.isdir(process_train_path) is False:
        os.mkdir(process_train_path)
    if op.isdir(process_test_path) is False:
        os.mkdir(process_test_path)

    audio_filenames = glob.glob(f'{folder_dir}/*.wav')
    num_files = len(audio_filenames)
    print(f'number of files in {folder_dir} is {num_files}')

    # remove the waveform less than two seconds.
    for audio in tqdm(audio_filenames):
        raw_audio = tf.io.read_file(audio)
        waveform, sr = tf.audio.decode_wav(raw_audio)
        if sr != hp.sr:
            raise ValueError("Unexpected sample rate:", sr)
        if waveform.shape[-1] != 1:
            raise ValueError("Unexpected channel with shape:", waveform.shape, audio)

        waveform = tf.reshape(waveform, waveform.shape[0])
        # Trim the silence at begin, at end.
        trim, interval = librosa.effects.trim(waveform)
        # Keep the waveform longer than expected length.
        if trim.shape[0]/sr < hp.min_length:
            os.remove(audio)
        else:
            if trim.shape[0] < waveform.shape[0]:
                trim = tf.reshape(trim, [trim.shape[0], 1])
                raw_wav = tf.audio.encode_wav(trim, sr)
                tf.io.write_file(audio, raw_wav)
            elif trim.shape[0] > waveform.shape[0]:
                raise ValueError("Unexpected error with trimmed longer than origin.:", trim.shape[0], waveform.shape[1])

    audio_filenames = glob.glob(f'{folder_dir}/*.wav')
    num_files = len(audio_filenames)
    print(f'After: number of files in {folder_dir} is {num_files}')

    # Split to train and test part.
    test_split_nun = int(_NUM_TEST * num_files)
    print(f"test file numbers: {test_split_nun}")

    random.seed(1373)
    random.shuffle(audio_filenames)
    train_files = audio_filenames[test_split_nun:]
    test_files = audio_filenames[:test_split_nun]

    for f in tqdm(train_files):
        dst = op.join(train_path, op.basename(f))
        copyfile(f, dst)
    for f in tqdm(test_files):
        dst = op.join(test_path, op.basename(f))
        copyfile(f, dst)

    # Extract features
    for audio_pth in [train_path, test_path]:
        for filename in tqdm(Path(audio_pth).glob('*.wav')):
            filename = str(filename)
            raw_audio = tf.io.read_file(filename)
            waveform, sr = tf.audio.decode_wav(raw_audio)
            if sr != hp.sr:
                raise ValueError("Unexpected sample rate:", sr)

            # waveform shape: [T, 1] -> [T]
            waveform = tf.reshape(waveform, waveform.shape[0])

            # mag shape: [T, n_fft//2]
            spec = toSpec_db_norm(waveform)

            # mel shape: [T, n_mels]
            de_spec = spec_denorm_deDB(spec)
            mel = toMel_db_norm(de_spec, sr)

            # Save to file
            basename = op.basename(filename).split('.')[0]
            dir_name = op.join(op.dirname(filename), process_dir)

            # wav shape: [T]
            wav_name = basename + hp.wav_suffix
            wav_name = op.join(dir_name, wav_name)
            np.save(wav_name, waveform)

            # mag shape: [T, n_fft//2] -> [n_fft//2, T]
            spec = tf.transpose(spec, perm=[1, 0])
            spec_name = basename + hp.mag_suffix
            spec_name = op.join(dir_name, spec_name)
            np.save(spec_name, spec)

            img_spec_name = spec_name + hp.img_ext
            plt.imsave(img_spec_name, spec, cmap='gray')

            mel_name = op.basename(filename).split('.')[0] + hp.mel_suffix
            mel_name = op.join(dir_name, mel_name)
            np.save(mel_name, mel)

            img_mel_name = mel_name + hp.img_ext
            plt.imsave(img_mel_name, mel, cmap='gray')
