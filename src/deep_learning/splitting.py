import pandas as pd
import numpy as np
from src.util import make_image
from src.data_representations.process_wavs import butter_highpass_filter
from src.data_representations.process_images import extract_melspecs


def create_txt(rec_labels, th=0.6, min_size=300, sr=22050):
    """ Creates txt file from rec_labels assigned to each sample"""
    df_txt = []
    prev_ind = -1
    current_window = None
    in_sound = False
    for ind, lbl in enumerate(rec_labels):
        if lbl >= th:
            if in_sound: 
                current_window = (current_window[0], ind)
            else:
                in_sound = True
                current_window = (ind, ind + 1)
        else:
            if in_sound:
                in_sound = False
                if current_window[1] - current_window[0] >= min_size:
                    df_txt.append(current_window)
            else:
                continue
    df_txt = pd.DataFrame(df_txt, columns=['s', 'e']) / sr
    df_txt['cl'] = 'sound'
    return df_txt


def broaden_timestamps(df_txt, rec, each_side=0.17, std=0.01, sr=22050):
    """ Broadens sounds timeframes from both sides """
    df_txt_broaden = []
    L_sec = len(rec) / sr
    last_ind = len(df_txt) - 1
    for ind in range(len(df_txt)):
        
        s, e = df_txt[['s', 'e']].iloc[ind]
        size = e - s
        
        max_s_shift = s if ind == 0 else (s - df_txt['s'].iloc[ind - 1])
        max_e_shift = (L_sec - e)if ind == last_ind else (df_txt['e'].iloc[ind + 1] - e)
        
        s_shift = min(max(0, np.random.normal(loc=size * each_side, scale=std)), max_s_shift)
        e_shift = min(max(0, np.random.normal(loc=size * each_side, scale=std)), max_e_shift)
        
        df_txt_broaden.append((s - s_shift,  e + e_shift, 'sound'))
        
    return pd.DataFrame(df_txt_broaden, columns=['s', 'e', 'cl'])


def split_recording(rec, model, th=0.75, n_fft=1024, n_mel=80, resolution=1024, step_size=256,
                    filtering_th=3000, broaden_factor=0.17, sr=22050):
    """ Splits recording with keras model """
    rec = butter_highpass_filter(rec, filtering_th, fs=sr)
    img = make_image(rec)
    
    predictions_for_each_pixel = resolution / step_size
    px_to_smp = (len(rec) / img.shape[1])
    sec_to_px = sr / (len(rec) / img.shape[1])
    
    sounds = [rec[s: s + resolution] for s in range(0, len(rec) - resolution + 1, step_size)]
    melspecs = np.array(extract_melspecs(sounds, n_fft, n_mel))
    predictions = model.predict(np.reshape(melspecs, (*np.shape(melspecs), 1)))
    
    rec_labels = np.zeros(len(rec))
    for sound_pr, ind in zip(predictions, range(0, len(rec) - resolution + 1, step_size)):
        rec_labels[ind: ind + resolution] += sound_pr[1] / predictions_for_each_pixel

    img_labels = []
    for ind in np.arange(0, len(rec) - px_to_smp, px_to_smp):
        pixel_label = np.mean(rec_labels[int(ind) : int(ind + px_to_smp)])
        img_labels.append(pixel_label)
    img_labels = np.array(img_labels)
    
    df_txt = create_txt(rec_labels, th, sr=sr)
    df_txt = broaden_timestamps(df_txt, rec, each_side=broaden_factor, sr=sr)
    return df_txt, img, rec_labels, img_labels
