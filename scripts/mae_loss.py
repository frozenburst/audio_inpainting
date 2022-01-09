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

import numpy as np
import tensorflow as tf
import os.path as op


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

    if op.isfile(output_pth) is False:
        raise ValueError("Output file should exist with test file.")

    loss_type = ['mae', 'w_mae', 'psnr', 'ssim']
    mae_list = []
    w_mae_list = []
    psnr_list = []
    ssim_list = []
    # Measure loss of output

    ref_spec = np.load(ref_spec_pth)
    ref_spec = tf.convert_to_tensor(ref_spec, tf.float32)
    ref_spec = tf.reshape(ref_spec, [1]+ref_spec.shape+[1])

    inpaint_spec = np.load(output_pth)
    inpaint_spec = tf.convert_to_tensor(inpaint_spec, tf.float32)
    if len(inpaint_spec.shape) == 3:
        inpaint_spec = tf.reshape(inpaint_spec, [1]+inpaint_spec.shape)
    elif len(inpaint_spec.shape) == 2:
        inpaint_spec = tf.reshape(inpaint_spec, [1]+inpaint_spec.shape+[1])

    if ref_spec.shape != inpaint_spec.shape:
        raise ValueError("Mismatch of spec shape:", ref_spec.shape, inpaint_spec.shape)

    mae_loss = mean_l1_loss(ref_spec, inpaint_spec)
    w_mae_loss = weighted_mae(ref_spec, inpaint_spec)
    psnr_loss = psnr(ref_spec, inpaint_spec)[0]
    ssim_loss = ssim(ref_spec, inpaint_spec)[0]

    print(f'MAE loss: {mae_loss}')
    print(f'W_MAE loss: {w_mae_loss}')
    print(f'PSNR loss: {psnr_loss}')
    print(f'SSIM loss: {ssim_loss}')
