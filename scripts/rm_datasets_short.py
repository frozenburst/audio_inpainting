from tqdm import tqdm

import tensorflow as tf
import os.path as op
import os
import glob
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--audio_path', default='/work/r08922a13/datasets/maestro-v3.0.0/sr41k/audio', type=str,
                    help='The place for audio')
parser.add_argument('--train_path', default='/work/r08922a13/datasets/maestro-v3.0.0/sr41k/train', type=str,
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


if __name__ == "__main__":
    args = parser.parse_args()
    audio_path = args.audio_path
    train_path = args.train_path
    test_path = args.test_path

    process_dir = hp.preprocess_dir
    process_train_path = op.join(train_path, process_dir)
    process_test_path = op.join(test_path, process_dir)

    for wav_path in [train_path, test_path]:
        audio_filenames = glob.glob(f'{wav_path}/*.wav')
        num_files = len(audio_filenames)
        print(f'number of files in {wav_path} is {num_files}')

        input(f"Sure to remove in path: {wav_path}? (Ctrl-c to exit.)")

        num_remove = 0
        for audio in tqdm(audio_filenames):
            raw_audio = tf.io.read_file(audio)
            waveform, sr = tf.audio.decode_wav(raw_audio)
            if sr != hp.sr:
                raise ValueError("Unexpected sample rate:", sr, audio)

            if waveform.shape[0]/sr < hp.min_length:
                audio_names = op.basename(audio).split('.')[0]
                # path = op.join(wav_path, process_dir)
                path = wav_path
                filenames = op.join(path, audio_names)
                remove_files = glob.glob(f'{filenames}*')
                for remove_file in remove_files:
                    os.remove(remove_file)
                num_remove += 1
                print(waveform.shape, num_remove)
            elif waveform.shape[0]/sr > hp.max_length:
                raise ValueError("Unexpected error with longer than max settings.:", waveform.shape)
