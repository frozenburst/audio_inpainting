#!/usr/bin/env python
# coding: utf-8
from tqdm import tqdm

import tensorflow as tf
import numpy as np
import os.path as op
import matplotlib.pyplot as plt
import os
import glob
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='/your/workspace/datasets/LJSpeech-1.1/wavs', type=str,
                    help='The data root')


class hp:
    sr = 44100      # Sampling rate. LJS:22050
    n_fft = 510     # let height of spec be 256
    win_length = n_fft
    hop_length = 256    # fix due to 510/2 = 255
    n_mels = 80
    power = 2
    max_db = 100
    ref_db = 20
    least_amp = 1e-5
    min_length = 5      # spec length 256 ~= 3 sec in sr=22050
    image_width = 256
    seg_middle = round(sr / hop_length * 3.5)
    seg_start = seg_middle - (image_width // 2)
    preprocess_dir = "preprocess"
    seg_dir = "seg"
    wav_dir = "wav"
    wav_suffix = "-wave"
    mag_suffix = "-mag-raw-feats"
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

    if op.isdir(folder_dir) is False:
        raise ValueError("The data folder is not exist with path:", folder_dir)

    process_path = op.join(folder_dir, hp.preprocess_dir)
    if op.isdir(process_path) is False:
        os.mkdir(process_path)

    seg_dir = op.join(process_path, hp.seg_dir)
    if op.isdir(seg_dir) is False:
        os.mkdir(seg_dir)

    wav_dir = op.join(seg_dir, hp.wav_dir)
    if op.isdir(wav_dir) is False:
        os.mkdir(wav_dir)

    audio_filenames = glob.glob(f'{folder_dir}/*.wav')
    num_files = len(audio_filenames)
    print(f'number of files in {folder_dir} is {num_files}')

    # Extract features
    for filename in tqdm(audio_filenames):
        raw_audio = tf.io.read_file(filename)
        waveform, sr = tf.audio.decode_wav(raw_audio)
        if sr != hp.sr:
            raise ValueError("Unexpected sample rate:", sr)

        seg_wave_name = op.join(wav_dir, op.basename(filename))
        seg_wave = waveform[:sr*5, :]
        seg_wav = tf.audio.encode_wav(seg_wave, sr)
        tf.io.write_file(seg_wave_name, seg_wav)

        # waveform shape: [T, 1] -> [T]
        waveform = tf.reshape(waveform, waveform.shape[0])

        # mag shape: [T, n_fft//2]
        spec = toSpec_db_norm(waveform)

        # Save to file
        basename = op.basename(filename)[:-4]

        # mag shape: [T, n_fft//2] -> [n_fft//2, T]
        spec = tf.transpose(spec, perm=[1, 0])
        spec_name = basename + hp.mag_suffix
        spec_name = op.join(process_path, spec_name)
        np.save(spec_name, spec)

        img_spec_name = spec_name + hp.img_ext
        plt.imsave(img_spec_name, spec, cmap='gray')

        seg_spec = spec[:, hp.seg_start:hp.seg_start+hp.image_width]
        seg_spec_name = basename + '_seg'
        seg_spec_name = op.join(seg_dir, seg_spec_name)
        np.save(seg_spec_name, seg_spec)
        seg_img_spec_name = seg_spec_name + hp.img_ext
        plt.imsave(seg_img_spec_name, seg_spec, cmap='gray')
