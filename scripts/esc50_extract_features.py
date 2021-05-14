#!/usr/bin/env python
# coding: utf-8
from pathlib import Path
from tqdm import tqdm

import tensorflow as tf
import numpy as np
import os
import os.path as op
import librosa
import math
#import torchaudio
#import torchaudio.transforms as transforms
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--train_path', default='/work/r08922a13/datasets/ESC-50-master/split/train', type=str,
                    help='The place for training data')
parser.add_argument('--test_path', default='/work/r08922a13/datasets/ESC-50-master/split/test', type=str,
                    help='The place for testing data.')


class hp:
    sr = 44100  # Sampling rate.
    n_fft = 510     # let height of spec be 256
    win_length = n_fft
    # hop_length = win_length // 2    # 256
    hop_length = 256    # fix due to 510/2 = 255
    n_mels = 80
    power = 2
    max_db = 100
    ref_db = 20
    least_amp = 1e-5
    min_length = 5
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
    #spec = transforms.Spectrogram(
    #    n_fft=hp.n_fft, win_length=hp.win_length, hop_length=hp.hop_length,
    #    power=hp.power, normalized=False)(waveform)
    # To decible
    #db_spec = 20 * np.log10(np.maximum(hp.least_amp, spec))
    # Normalize
    #db_spec_norm = np.clip(db_spec / hp.max_db, -1, 1)
    return db_spec_norm


def spec_denorm_deDB(spec):
    # De-normalize
    de_norm_spec = spec * hp.max_db
    # To amplitude
    #de_db_spec = np.power(10.0, de_norm_spec * 0.05)
    de_db_spec = tf.math.pow(10.0, de_norm_spec * 0.05)
    return de_db_spec


def toMel_db_norm(deDB_denorm_spec, sr):
    '''
    mel = transforms.MelSpectrogram(
         n_mels=hp.n_mels, n_fft=hp.n_fft, win_length=hp.win_length, hop_length=hp.hop_length,
         power=hp.power, normalized=False)(waveform)
    '''
    #mel = transforms.MelScale(n_mels=hp.n_mels, sample_rate=sr)(deDB_denorm_spec)
    to_mel_m = tf.signal.linear_to_mel_weight_matrix(hp.n_mels, deDB_denorm_spec.shape[-1], sr)
    mel = tf.tensordot(deDB_denorm_spec, to_mel_m, 1)
    # To decible
    #db_mel = 20 * np.log10(np.maximum(hp.least_amp, mel))
    db_mel = hp.ref_db * tf.experimental.numpy.log10(tf.math.maximum(hp.least_amp, mel))
    # Normalize
    #db_mel_norm = np.clip(db_mel / hp.max_db, -1, 1)
    db_mel_norm = tf.clip_by_value(db_mel / hp.max_db, -1, 1)
    return db_mel_norm


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    train_path = args.train_path
    test_path = args.test_path

    if op.isdir(train_path) is False:
        raise ValueError("The data folder is not exist with path:", train_path)
    if op.isdir(test_path) is False:
        raise ValueError("The data folder is not exist with path:", test_path)

    process_dir = 'tf_preprocess'
    process_train_path = op.join(train_path, process_dir)
    process_test_path = op.join(test_path, process_dir)
    if op.isdir(process_train_path) is False:
        os.mkdir(process_train_path)
    if op.isdir(process_test_path) is False:
        os.mkdir(process_test_path)

    for audio_pth in [train_path, test_path]:
        for filename in tqdm(Path(audio_pth).glob('*.wav')):
            # print(filename)
            filename = str(filename)
            #waveform, sr = torchaudio.load(filename, normalize=True)
            raw_audio = tf.io.read_file(filename)
            waveform, sr = tf.audio.decode_wav(raw_audio)
            # waveform shape: [T, 1] -> [T]
            waveform = tf.reshape(waveform, waveform.shape[0])

            if sr != hp.sr:
                raise ValueError("Unexpected sample rate:", sr)

            # Trim the silence at begin, at end.
            # Pad to 5 sec.
            trim, interval = librosa.effects.trim(waveform)
            if trim.shape[0]/sr < hp.min_length:
                n = int(tf.math.ceil(hp.min_length*sr / trim.shape[0]))
                trim = np.tile(trim, (n))[:hp.min_length * sr]
                # trim = torch.tensor(trim)
                # trim = trim[np.newaxis, :]

            # patch_length = int(hp.n_fft/2 + 1)
            # mag shape: [T, n_fft//2]
            spec = toSpec_db_norm(trim)

            # mel shape: [T, n_mels]
            de_spec = spec_denorm_deDB(spec)
            mel = toMel_db_norm(de_spec, sr)

            # Save to file
            basename = op.basename(filename).split('.')[0]
            dir_name = op.join(op.dirname(filename), process_dir)

            # wav shape: [T]
            wav_name = basename + hp.wav_suffix
            wav_name = op.join(dir_name, wav_name)
            np.save(wav_name, trim)

            wav_name = op.basename(filename)
            wav_name = op.join(dir_name, wav_name)
            trim = tf.reshape(trim, [trim.shape[0], 1])
            raw_wav = tf.audio.encode_wav(trim, sr)
            tf.io.write_file(wav_name, raw_wav)
            #trim = trim[np.newaxis, :]
            #torchaudio.save(filepath=wav_name, src=trim, sample_rate=sr)

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
