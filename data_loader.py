import numpy as np

def load_data_filename(filename, batch=False):
    datas = []
    labels = []
    with open(filename, 'r') as f:
        data_list = f.read().splitlines()
        for line in data_list:
            #datas.append(line.split(' ')[0])
            data_name = line.split(' ')[0]
            data = np.load(data_name)
            if batch is False:
                w, h, _ = data.shape
                data = data.reshape(w, h)
            datas.append(data)

            labels.append(int(line.split(' ')[1]))
    return np.array(datas), np.array(labels)

def get_class_name():
    class_names = ['dog', 'rooster', 'pig', 'cow', 'frog',
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
