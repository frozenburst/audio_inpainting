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
import math
import librosa
import glob
import random
import argparse
#import torchaudio
#import torchaudio.transforms as transforms

# 1/8 to make sure balanced data.
_NUM_TEST = 0.1
_NUM_CLASS = 0
_NUM_PART = 0
parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='/work/r08922a13/datasets/maestro-v3.0.0', type=str,
                    help='The data root')
parser.add_argument('--audio_path', default='/work/r08922a13/datasets/maestro-v3.0.0/sr41k/audio', type=str,
                    help='The place for audio')
parser.add_argument('--train_path', default='/work/r08922a13/datasets//maestro-v3.0.0/sr41k/train', type=str,
                    help='The place for training data')
parser.add_argument('--test_path', default='/work/r08922a13/datasets/maestro-v3.0.0/sr41k/test', type=str,
                    help='The place for testing data.')


class hp:
    sr = 44100  # Sampling rate. maestro-v3.0.0
    n_fft = 510     # let height of spec be 256 (nfft/2 +1)
    win_length = n_fft
    # hop_length = win_length // 2    # 256
    hop_length = 256    # fix due to 510/2 = 255
    n_mels = 80
    power = 2
    max_db = 100
    ref_db = 20.0
    least_amp = 1e-5
    min_length = 5      # spec length 256 ~= 1.5 sec in sr=44100
    max_length = 10
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
    audio_path = args.audio_path
    train_path = args.train_path
    test_path = args.test_path

    if op.isdir(folder_dir) is False:
        raise ValueError("The data folder is not exist with path:", folder_dir)
    # mask dir of sr41
    if op.isdir(op.dirname(audio_path)) is False:
        os.mkdir(op.dirname(audio_path))
    if op.isdir(audio_path) is False:
        os.mkdir(audio_path)
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

    audio_filenames = glob.glob(f'{folder_dir}/20*/*.wav')
    num_files = len(audio_filenames)
    print(f'number of files in {folder_dir} is {num_files}')

    stage_chopped = input("Chopped the maestro into pieces? (y or n(skip)):")

    num_unexpected_sr = 0
    if stage_chopped == 'y':
        for audio in tqdm(audio_filenames):
            #waveform, sr = torchaudio.load(audio, normalize=True)
            raw_audio = tf.io.read_file(audio)
            waveform, sr = tf.audio.decode_wav(raw_audio)
            if sr != hp.sr:
                #raise ValueError("Unexpected sample rate:", sr, audio)
                num_unexpected_sr += 1
                print("Unexpected sample rate:", sr, audio, num_unexpected_sr)
                continue
            # Maestro should be stereo
            if waveform.shape[-1] != 2:
                raise ValueError(f"Unexpected channel for maestro: {waveform.shape}")
            # split stereo to single channel.
            waveform_track1, waveform_track2 = tf.split(waveform, 2, axis=1)
            for i, waveform in enumerate([waveform_track1, waveform_track2]):
                waveform = tf.reshape(waveform, waveform.shape[0])
                # Trim the silence at begin, at end.
                trim, interval = librosa.effects.trim(waveform)

                # We don't consider the segments < 10sec, which should not appear in maestro.
                if trim.shape[0]/sr > hp.max_length:
                    num_parts = math.ceil((trim.shape[0]/sr) / hp.max_length)
                    for part in range(num_parts):
                        if part != (num_parts - 1):
                            trim_part = trim[part*hp.max_length*sr: (part+1)*hp.max_length*sr]
                        # Last part might be shorter.
                        else:
                            trim_part = trim[part*hp.max_length*sr:]
                        trim_part_trim, _ = librosa.effects.trim(trim_part)
                        if trim_part_trim.shape[0] > hp.min_length:
                            trim_part_trim = tf.reshape(trim_part_trim, [trim_part_trim.shape[0], 1])
                            raw_wav = tf.audio.encode_wav(trim_part_trim, sr)
                            basename = op.basename(audio).split('.')[0]
                            audio_name = f'{basename}_track{i}_{part}{hp.wav_ext}'
                            path = op.join(audio_path, audio_name)
                            tf.io.write_file(path, raw_wav)
                        elif trim_part_trim.shape[0] > hp.max_length:
                            raise ValueError("Unexpected error with trimmed longer than max settings.:", trim_part_trim.shape[0])
    elif stage_chopped != 'n':
        raise ValueError(f"Unrecognized input with value: {stage_chopped}. Stop the program.")

    audio_filenames = glob.glob(f'{audio_path}/*.wav')
    num_files = len(audio_filenames)
    print(f'After chopped: number of files in {audio_path} is {num_files}')

    # Split to train and test part.
    test_split_nun = int(_NUM_TEST * num_files)
    print(f"test file numbers: {test_split_nun}")

    random.seed(1373)
    random.shuffle(audio_filenames)
    train_files = audio_filenames[test_split_nun:]
    test_files = audio_filenames[:test_split_nun]

    stage_split = input("Ready to split into train test, make sure storage is enough about 200GB+. y(yes) or n(skip).")

    if stage_split == 'y':
        for f in tqdm(train_files):
            dst = op.join(train_path, op.basename(f))
            copyfile(f, dst)
        for f in tqdm(test_files):
            dst = op.join(test_path, op.basename(f))
            copyfile(f, dst)
    elif stage_split != 'n':
        raise ValueError("Unexpected commends:", stage_split)

    stage_extract = input("Ready to extract feature. y or n?")
    if stage_extract != 'y':
        print("Skip stage of extract feature.")
        exit()
    # Extract features
    for audio_pth in [train_path, test_path]:
        for filename in tqdm(Path(audio_pth).glob('*.wav')):
            #waveform, sr = torchaudio.load(filename, normalize=True)
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
