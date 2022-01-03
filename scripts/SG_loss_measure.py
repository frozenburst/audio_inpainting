#!/usr/bin/env python
# coding: utf-8
"""
Test the generative model with own settings.

usage: batch_test_loss_measure.py  [options]

options:
    --ref_spec_pth=<ref_spec path>
    --output_pth=<output path>
"""
from docopt import docopt
from pathlib import Path
from tqdm import tqdm

import numpy as np
import tensorflow as tf
import os.path as op
import os


def l1_loss(x, y):
    return tf.math.reduce_mean(tf.math.abs(x - y))


def mean_l1_loss(x, y):
    error = tf.math.abs(x - y)
    return tf.math.reduce_mean(error)


def weighted_mae(x, y):
    loss_matrix = tf.math.abs(x - y)
    _, H, W, _ = loss_matrix.shape
    w = tf.linspace(tf.ones(W), tf.ones(W)*10.0, H, axis=0)
    w = 1.-tf.experimental.numpy.log10(w)
    w = tf.reshape(w, [1, H, W, 1])
    ones_w = tf.ones_like(w)
    scaler = tf.reduce_sum(ones_w) / tf.reduce_sum(w)
    w_mae_matrix = loss_matrix * w * scaler
    return tf.math.reduce_mean(w_mae_matrix)


def psnr(target, ref):
    return tf.image.psnr(target, ref, max_val=1.0)


def ssim(target, ref):
    return tf.image.ssim(target, ref, max_val=1.0)


if __name__ == "__main__":
    args = docopt(__doc__)
    print(args)
    ref_spec_pth = args['--ref_spec_pth']
    output_pth = args['--output_pth']

    if output_pth is None:
        raise ValueError("Please set the path for model's output.")
    if ref_spec_pth is None:
        raise ValueError("Path of reference spec not set!")

    if op.isdir(output_pth) is False:
        raise ValueError("Output path should exist with batch test flist.")

    loss_type = ['mae', 'w_mae', 'psnr', 'ssim']
    mae_list = []
    w_mae_list = []
    psnr_list = []
    ssim_list = []

    ext = '-mag-raw-feats.npy'
    # Measure loss of output
    for i, filename in tqdm(enumerate(sorted(Path(output_pth).glob('*rec-mag-raw-feats.npy')))):
        file_basename = op.basename(filename).split('_')[0] + ext

        ref_spec_filename = op.join(ref_spec_pth, file_basename)
        ref_spec = np.load(ref_spec_filename)
        ref_spec = tf.convert_to_tensor(ref_spec, tf.float32)
        ref_spec = tf.reshape(ref_spec, [1]+ref_spec.shape+[1])

        inpaint_spec = np.load(filename)
        inpaint_spec = tf.convert_to_tensor(inpaint_spec, tf.float32)
        inpaint_spec = tf.reshape(inpaint_spec, [1]+inpaint_spec.shape+[1])

        if ref_spec.shape != inpaint_spec.shape:
            raise ValueError("Mismatch of spec shape:", ref_spec.shape, inpaint_spec.shape)

        mae_list.append(mean_l1_loss(ref_spec, inpaint_spec))
        w_mae_list.append(weighted_mae(ref_spec, inpaint_spec))
        psnr_list.append(psnr(ref_spec, inpaint_spec)[0])
        ssim_list.append(ssim(ref_spec, inpaint_spec)[0])

    # If the diretory of loss not exist, create it.
    loss_pth = op.join(output_pth, 'loss')
    if op.isdir(loss_pth) is False:
        os.mkdir(loss_pth)
    for loss in loss_type:
        loss_filename = op.join(loss_pth, f'{loss}_loss.txt')
        if loss == 'mae':
            loss_list = mae_list
        elif loss == 'w_mae':
            loss_list = w_mae_list
        elif loss == 'psnr':
            loss_list = psnr_list
        elif loss == 'ssim':
            loss_list = ssim_list

        with open(loss_filename, 'w') as f:
            for content in sorted(loss_list):
                f.write(f'{content}\n')
        print("Save loss file to:", loss_filename)
