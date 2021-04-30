import numpy as np


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
