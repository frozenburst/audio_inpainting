from tqdm import tqdm
import numpy as np


def random_crop_img(img, length=256):
    rand_start = np.random.randint(img.shape[1] - length)
    return img[:, rand_start:rand_start+length]


def load_npy(filename, max_length=862):
    # H * T: [256, T]
    data = np.load(filename)
    # [256, T=5sec]
    data = data[:, :, np.newaxis]

    assert data.shape[1] >= max_length, f"data is too short with shape: {data.shape}"

    if data.shape[1] > max_length:
        data = random_crop_img(data, max_length)

    return data


def load_npy_test(filename, max_length=862):
    # H * T: [256, T]
    data = np.load(filename)
    # [256, T=5sec]
    data = data[:, :, np.newaxis]

    assert data.shape[1] >= max_length, f"data is too short with shape: {data.shape}"

    if data.shape[1] > max_length:
        # data = random_crop_img(data, max_length)
        data = data[:, :max_length]

    return data, filename


def load_data_filename(filename, islabeled=False):
    data_fnames = []
    labels = []
    if islabeled:
        with open(filename, 'r') as f:
            data_list = f.read().splitlines()
            for line in tqdm(data_list):
                data_name = line.split(' ')[0]
                data_fnames.append(data_name)
                labels.append(int(line.split(' ')[1]))
    else:
        with open(filename, 'r') as f:
            data_fnames = f.read().splitlines()

    return np.array(data_fnames), np.array(labels)


def get_class_name():
    class_names = [
          'dog', 'rooster', 'pig', 'cow', 'frog',
          'cat', 'hen', 'insects', 'sheep', 'crow',
          'rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds',
          'water_drops', 'wind', 'pouring_water', 'toilet_flush', 'thunderstorm',
          'crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing',
          'footsteps', 'laughing', 'brushing_teeth', 'snoring', 'drinking_sipping',
          'door_wood_knock', 'mouse_click', 'keyboard_typing', 'door_wood_creaks', 'can_opening',
          'washing_machine', 'vacuum_cleaner', 'clock_alarm', 'clock_tick', 'glass_breaking',
          'helicopter', 'chainsaw', 'siren', 'car_horn', 'engine',
          'train', 'church_bells', 'airplane', 'fireworks', 'hand_saw']
    return class_names
