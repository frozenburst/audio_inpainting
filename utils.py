import tensorflow as tf
import numpy as np
import librosa


class hp:
    sr = 44100  # Sampling rate.
    n_fft = 510
    n_mels = 80
    win_length = n_fft
    hop_length = 256
    power = 2
    max_db = 100
    ref_db = 20.0
    least_amp = 1e-5


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


def denorm_toMel_db_norm(specs):
    deDB_denorm_spec = spec_denorm_deDB(specs)
    to_mel_m = tf.signal.linear_to_mel_weight_matrix(hp.n_mels, deDB_denorm_spec.shape[-1], hp.sr)
    mel = tf.tensordot(deDB_denorm_spec, to_mel_m, 1)
    # To decible
    db_mel = hp.ref_db * tf.experimental.numpy.log10(tf.math.maximum(hp.least_amp, mel))
    # Normalize
    db_mel_norm = tf.clip_by_value(db_mel / hp.max_db, -1, 1)
    return db_mel_norm


def mag_to_mel(specs):
    if len(specs.shape) == 4:
        b, h, w, c = specs.shape
        specs = tf.reshape(specs, [b, h, w])
        # B * bins * T -> B * T * bins
        specs = tf.transpose(specs, perm=[0, 2, 1])
        mels = denorm_toMel_db_norm(specs)
        return mels
    elif len(specs.shape) == 2:
        h, w = specs.shape
        specs = tf.reshape(specs, [1, h, w])
        specs = tf.transpose(specs, perm=[0, 2, 1])
        mels = denorm_toMel_db_norm(specs)
        return mels
    else:
        raise ValueError("The shapes is not as expected:", specs.shape)


# Reference from GitHub: https://github.com/taki0112/SPADE-Tensorflow
def pytorch_xavier_weight_factor(gain=0.02, uniform=False) :

    if uniform :
        factor = gain * gain
        mode = 'fan_avg'
    else :
        factor = (gain * gain) / 1.3
        mode = 'fan_avg'

    return factor, mode, uniform

def pytorch_kaiming_weight_factor(a=0.0, activation_function='relu', uniform=False) :

    if activation_function == 'relu' :
        gain = np.sqrt(2.0)
    elif activation_function == 'leaky_relu' :
        gain = np.sqrt(2.0 / (1 + a ** 2))
    elif activation_function == 'tanh' :
        gain = 5.0 / 3
    else :
        gain = 1.0

    if uniform :
        factor = gain * gain
        mode = 'FAN_IN'
    else :
        factor = (gain * gain) / 1.3
        mode = 'FAN_IN'

    return factor, mode, uniform



