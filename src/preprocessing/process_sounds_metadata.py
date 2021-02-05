from src.util import date_to_datetime
import numpy as np
import pandas as pd


def compute_epochs(sounds_metadata, EPOCHS):
    """ Generates sounds_metadata with additional epochs column """
    def _get_epoch(dt, epochs_dict):
        if epochs_dict is None:
            return '?'
        for s, e in epochs_dict:
            if date_to_datetime(s) <= dt <= date_to_datetime(e):
                return epochs_dict[(s, e)]
    epochs = []
    for ind in range(len(sounds_metadata)):
        col = sounds_metadata['colony'].iloc[ind]
        date_dt = date_to_datetime(sounds_metadata['date'].iloc[ind])
        epochs.append(_get_epoch(date_dt, EPOCHS[col]))
    sounds_metadata['epoch'] = epochs
    return sounds_metadata


def generate_sounds_metadata(recs_metadata, sounds_to_exclude = [],
                             columns_to_copy = ['colony', 'ratids', 'date', 'experiment']):
    """ Generates metadata table for all sounds from all recordings that are not fresh 
        Copies columns_to_copy columns from recordings metadata into sounds_metadata  """
    processed_recs = recs_metadata[recs_metadata['processing stage'] != 'fresh']
    sounds_metadata = []
    for p, n in processed_recs[['path', 'name']].values:
        rec_row = recs_metadata[recs_metadata['name'] == n]
        df_txt = pd.read_csv(p + n[:-3] + 'txt', sep = '\t')
        df_txt['rec'] = n
        for c in columns_to_copy:
            df_txt[c] = rec_row[c].iloc[0]
        sounds_metadata.append(df_txt[~df_txt['cl'].isin(sounds_to_exclude)])
    sounds_metadata = pd.concat(sounds_metadata, 0)
    return sounds_metadata

def extend_sounds_metadata(sounds_metadata, info):
    """ Extends sounds metadata with ratinfo """
    each_rat_info = {}
    for ratid in sounds_metadata['ratids'].unique():
        if '_' in ratid:
            w = s = r = db = bl = bl2 = age = np.nan
        elif int(ratid) not in info['ID'].values:
            print('No info for %s' % ratid)
            w = s = r = db = bl = bl2 = age = np.nan
        else:
            ind_in_info = np.where(info['ID'] == int(ratid))[0][0]
            w, s, r, db, bl, bl2 = info[['weight', 'sex', 'rank', 'dob', 'body length', 'body length2']].iloc[ind_in_info]
        each_rat_info[ratid] =  (w, s, r, db, bl, bl2)
        
    columns_to_add = []
    for ind in range(sounds_metadata.shape[0]):
        w, s, r, db, bl, bl2 = each_rat_info[sounds_metadata['ratids'].iloc[ind]]
        date = sounds_metadata['date'].iloc[ind]
        age = (date_to_datetime(date) - date_to_datetime(db)).days if (db is not np.nan) else np.nan
        columns_to_add.append((w, s, r, db, age, bl, bl2))
        
    columns_to_add = pd.DataFrame(columns_to_add, columns = ['weight', 'sex', 'rank', 
                                                             'dob', 'age', 'bodylength', 'bodylength2'])
    return pd.concat([sounds_metadata.reset_index(drop = True), columns_to_add], 1)

def make_fixed_size_sounds(sounds_metadata, resolution = 1024, step = 512):
    """ Changes timesstamps s.t. sounds are all of the same size """
    sounds_metadata_split = []
    
    s_ints = sounds_metadata['s'].apply(lambda x : int(sr * x))
    e_ints = sounds_metadata['e'].apply(lambda x : int(sr * x))
    sizes = e_ints - s_ints
    
    s_col_ind = list(sounds_metadata.columns).index('s')
    e_col_ind = list(sounds_metadata.columns).index('e')
    
    for ind in range(len(sounds_metadata)):
        s_int, e_int, size = s_ints.iloc[ind], e_ints.iloc[ind], sizes.iloc[ind]
        
        useless_space = sound_size - parts_in_sound * resolution
        
        s_int_new = int(s_int + useless_space // 2)
        e_int_new = int(e_int - useless_space // 2)
        
        for s_p in range(s_int_new, e_int_new + 1 - resolution, step):
            e_p = s_p + step
            row = list(sounds_metadata.iloc[ind])
            row[s_col_ind] = s_p
            row[e_col_ind] = e_p
            sounds_metadata_split.append(tuple(row))
            
    return pd.DataFrame(sounds_metadata_split, columns = sounds_metadata.columns)
    
    
    