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
    ref_db = 20
    least_amp = 1e-5


def spec_denorm_deDB(spec):
    # De-normalize
    de_norm_spec = spec * hp.max_db
    # To amplitude
    de_db_spec = tf.math.pow(10.0, de_norm_spec * 0.05)
    return de_db_spec


def to_mel(spec):
    return librosa.feature.melspectrogram(sr=hp.sr, S=spec, n_mels=hp.n_mels, htk=True, norm=None)


def denorm_toMel_db_norm(specs):
    mels = tf.TensorArray(tf.float32, size=specs.shape[0])
    for i in range(specs.shape[0]):
        #if len(spec.shape) != 3:
        #    raise ValueError("The shapes is not as expected:", spec.shape)
        deDB_denorm_spec = spec_denorm_deDB(specs[i])
        #x, t, _ = deDB_denorm_spec.shape
        #deDB_denorm_spec = np.reshape(deDB_denorm_spec, [x, t])
        # mel = transforms.MelScale(n_mels=hp.n_mels, sample_rate=sr)(deDB_denorm_spec)
        # Same as Pytorch
        mel = tf.py_function(
            func=to_mel,
            inp=[deDB_denorm_spec],
            Tout=tf.float32)
        # To decible
        #db_mel = 20 * np.log10(np.maximum(hp.least_amp, mel))
        db_mel = 20.0 * tf.experimental.numpy.log10(tf.math.maximum(hp.least_amp, mel))
        # Normalize
        #db_mel_norm = np.clip(db_mel / hp.max_db, -1, 1)
        db_mel_norm = tf.clip_by_value(db_mel / hp.max_db, -1, 1)
        mels = mels.write(i, db_mel_norm)
        #mels.append(db_mel_norm)
    return mels.stack()


def mag_to_mel(specs):
    if len(specs.shape) != 4:
        raise ValueError("The shapes is not as expected:", specs.shape)
    else:
        b, h, w, c = specs.shape
        specs = tf.reshape(specs, [b, h, w])
        mels = denorm_toMel_db_norm(specs)
        #breakpoint()
        #specs.set_shape([b, h, w])
        #mels = tf.py_function(
        #    func=denorm_toMel_db_norm,
        #    inp=[specs],
        #    Tout=tf.float32)
        #mels = tf.convert_to_tensor(mels)
        mels.set_shape([b, hp.n_mels, w])
        #mels = tf.reshape(mels, [b, hp.n_mels, w])
        return mels


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



