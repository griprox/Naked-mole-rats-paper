from scipy import signal
import numpy as np
import librosa

def butter_highpass(cutoff, fs, order=5):
    """ Auxillary function for filtering """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs = 22050, order = 5):
    """ Filters recording with the highpass filter """
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y
    
def add_noise(rec, sig = 0.005):
    """ Adds random gaussian noise to the recording """
    if sig is None:
        return rec
    noise = np.random.randn(len(rec))
    return sig * noise + rec

def stretch(data, rate = 1):
    """ Stretches the recording"""
    if rate is None:
        return data
    data = librosa.effects.time_stretch(data, rate)
    return data

def process_waves(sounds_npy, stretching_rate_lim, noise_sigma_lim, filtering_th):
    """ Performs data augmentation methods on sounds from sounds_npy"""
    sounds_array_pr = []
    inds = list(range(0, len(sounds_npy)))
    
    if stretching_rate_lim is not None:
        stretching_rates = np.random.uniform(stretching_rate_lim[0], stretching_rate_lim[1], 
                                             size = len(sounds_npy)).tolist()
    else:
        stretching_rates = [None] * len(sounds_npy)
        
    if noise_sigma_lim is not None:
        noise_sigmas = np.random.uniform(noise_sigma_lim[0], noise_sigma_lim[1],
                                         size = len(sounds_npy)).tolist()
    else:
        noise_sigmas = [None] * len(sounds_npy)

    def _process_sound(ind):
        s_pr = stretch(sounds_npy[ind], stretching_rates[ind])
        s_pr = add_noise(s_pr, noise_sigmas[ind])
        s_pr = butter_highpass_filter(s_pr, filtering_th)
        return s_pr
        
    sounds_npy_pr = (map(_process_sound, inds))
    
        
    return sounds_npy_pr

        
    