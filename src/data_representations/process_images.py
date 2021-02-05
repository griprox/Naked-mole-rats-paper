from skimage.transform import resize
import numpy as np
import librosa

def extract_1dmfcc(sounds, n_mel = 40, nfft = None, ampt_to_db = None, **kwargs):
    ''' Compute 1d vector features of mean, var and median of mfcc's taken in time axis'''
    mfcc_list = []
    for s in sounds:
        mfcc = librosa.feature.mfcc(y = s, sr = 22050, n_mel = n_mel).T
        mfcc_mean = np.mean(mfcc, 0)
        mfcc_var = np.var(mfcc, 0)
        mfcc_median = np.median(mfcc, 0)
        mfcc_list.append(np.concatenate([mfcc_mean, mfcc_var, mfcc_median]))
    return np.array(mfcc_list)

def extract_specs(sounds, nfft, amp_to_db = True, n_mel = None, **kwargs):
    ''' Computes regular spectrogramms '''
    specs = []
    for s in sounds:
        D = np.abs(librosa.stft(s, n_fft = nfft))
        D = np.flip(D, 0)
        if amp_to_db:
            D = librosa.amplitude_to_db(D, ) 
        specs.append(D)
    return specs

def extract_melspecs(sounds, nfft, n_mel = 40, amp_to_db = True, **kwargs):
    ''' Computes mel spectrogramms '''
    
    mel_basis = librosa.filters.mel(sr = 22050, n_fft = nfft, n_mels = n_mel)
    
    def _proc_func(s):
        spec, _ = librosa.core.spectrum._spectrogram(y = s, n_fft = nfft, hop_length = int(nfft/2), power = 1)    
        return np.log(np.dot(mel_basis, spec))
        
    specs = [_proc_func(s) for s in sounds]
    return specs
    
def extract_mfccs(sounds, n_mel = 40, nfft = None, ampt_to_db = None, **kwargs):
    ''' Computes spectrogramm of mfcc's '''
    specs = []
    for s in sounds:
        mfccs = librosa.feature.mfcc(y = s, sr = 22050, n_mfcc = n_mel)
        specs.append(mfccs)
    return specs
    

    
def random_roll(im):
    ''' Randomly rolls image in both dimensions '''
    N = im.shape[1]
    shift_max = int(N / 12)
    
    shift = np.random.randint(-shift_max, shift_max + 1)
    im_s = np.roll(im, shift, axis = 1)
    N = im_s.shape[0]
    shift_max = int(N / 16)
    shift = np.random.randint(-shift_max, shift_max//2 + 1)
    return np.roll(im_s, shift, axis = 0)

def random_zoom(im):
    ''' Performs zoom with random rate '''
    rate = np.random.uniform(0, 0.15)
    a, b = im.shape
    a_max = int(rate * a)
    b_max = int(rate * b)
    
    a_s = np.random.randint(0, a_max + 1)
    b_s = np.random.randint(0, b_max + 1)
    
    a_e = -1 - np.random.randint(0, a_max + 1)
    b_e = b - np.random.randint(0, b_max + 1)
    
    im_zoom = im[a_s : a_e, b_s : b_e]
    return im_zoom

def random_time_mask(im):
    ''' Masks out random time band '''
    im_copy = np.copy(im)
    
    width = np.random.choice([0, 1, 2, 3], p = [0.5, 0.23, 0.23, 0.04])
    width = min(width, int(im.shape[1]//2))
    mask_start = np.random.randint(0, im.shape[1] - width)
    mask = np.zeros(im.shape, dtype = 'bool')
    mask[:, np.arange(mask_start, mask_start + width)] = True
    im_copy[mask] =  np.mean(im_copy[mask])
    
    return im_copy

def random_freq_mask(im):
    ''' Masks out random frequency band '''
    im_copy = np.copy(im)
    
    width = np.random.choice([0, 1, 2, 3], p = [0.5, 0.23, 0.23, 0.04])
    mask_start = np.random.randint(0, im.shape[0] - width)
    mask = np.zeros(im.shape, dtype = 'bool')
    mask[np.arange(mask_start, mask_start + width)] = True
    im_copy[mask] =  np.mean(im_copy[mask])
    
    return im_copy

def resize_with_padding(im, target_shape):
    ''' Transform image to targe_shape. To decrease dimension this funcitons crops image, to incresse -- pads with mean '''
    if target_shape is None:
        return im
    diff0 = target_shape[0] - im.shape[0]
    diff1 = target_shape[1] - im.shape[1]
    if diff0 == 0:
        im_rs = im
        
    elif diff0 < 0:
        start0 = np.random.randint(0, -diff0)
        end0 = -diff0 - start0
        im_rs = im[start0 : -end0]
    else:
        pad_top = np.random.randint(0, diff0)
        pad_bot = diff0 - pad_top
        im_rs = np.concatenate([np.zeros((pad_top, im.shape[1])), im, np.zeros((pad_bot, im.shape[1]))])
        
    if diff1 == 0:
        pass
        
    elif diff1 < 0:
        start1 = np.random.randint(0, -diff1)
        end1 = -diff1 - start1
        im_rs = im_rs[:, start1 : -end1]
    else:
        pad_left = np.random.randint(0, diff1)
        pad_right = diff1 - pad_left
        im_rs = np.concatenate([np.zeros((im_rs.shape[0], pad_left)), im_rs, np.zeros((im_rs.shape[0], pad_right))],
                               1)
    assert im_rs.shape == target_shape, im_rs.shape
    return im_rs

def augment_im(im, target_shape):
    return resize_with_padding(random_time_mask(random_freq_mask(random_zoom(random_roll(im)))), target_shape)