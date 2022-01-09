#!/usr/bin/env python
# coding: utf-8
"""
Transfer the output spectrogram to audio.

usage: 2audio.py  [options]

options:
    --spec_pth=<spectrogram file directory>
"""
from docopt import docopt
from pathlib import Path
from tqdm import tqdm

import numpy as np
import os.path as op
import os
import torch
import torchaudio


class hp:
    sr = 44100  # Sampling rate.
    n_fft = 510
    win_length = n_fft
    hop_length = 256
    power = 2
    max_db = 100
    ref_db = 20
    least_amp = 1e-5


def toAudio_2amp_denorm(db_spec_norm):
    de_db_spec_norm = db_spec_norm * hp.max_db
    de_db_spec = np.power(10.0, de_db_spec_norm * 0.05)
    re_waveform = torchaudio.transforms.GriffinLim(
        n_fft=hp.n_fft, win_length=hp.win_length, hop_length=hp.hop_length,
        power=hp.power, normalized=True)(de_db_spec)
    return re_waveform


if __name__ == "__main__":
    args = docopt(__doc__)
    print(args)
    spec_pth = args['--spec_pth']

    if spec_pth is None:
        raise ValueError("Please set the path for spectrogram directory.")

    output_audio_pth = op.join(spec_pth, 'griffin-lim')
    if op.isdir(output_audio_pth) is False:
        os.mkdir(output_audio_pth)
    for filename in tqdm(sorted(Path(spec_pth).glob('*-mag-raw-feats.npy'))):
        spec = np.load(filename)
        h, w = spec.shape
        spec = spec.reshape((1, h, w))
        spec = torch.from_numpy(spec)

        res_waveform = toAudio_2amp_denorm(spec)

        audio_name = op.basename(filename).split('.')[0] + '.wav'
        audio_name = op.join(output_audio_pth, audio_name)
        torchaudio.save(audio_name, src=res_waveform, sample_rate=hp.sr)
