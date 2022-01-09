#!/usr/bin/env python
# coding: utf-8
"""
Transfer the output spectrogram to audio.

usage: npy2png.py  [options]

options:
    --spec_pth=<spectrogram file directory>
"""
from docopt import docopt
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import os


if __name__ == "__main__":
    args = docopt(__doc__)
    print(args)
    spec_pth = args['--spec_pth']

    if spec_pth is None:
        raise ValueError("Please set the path for spectrogram directory.")

    png_pth = op.join(spec_pth, 'png')
    if op.isdir(png_pth) is False:
        os.mkdir(png_pth)
    print("Load npy of output spec and save as png file...")
    for filename in tqdm(sorted(Path(spec_pth).glob('*.npy'))):
        spec = np.load(filename)
        if len(spec.shape) == 3:
            h, w, _ = spec.shape
            spec = spec.reshape((h, w))

        png_filename = op.basename(filename).split('.')[0] + '.png'
        png_filename = op.join(png_pth, png_filename)

        # First save the spec to png with force try.
        plt.imsave(png_filename, spec, cmap='gray')
