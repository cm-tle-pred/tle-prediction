import pandas as pd
import numpy as np
from sgp4.api import Satrec, WGS72
from datetime import datetime
from tqdm import tqdm
import os

def jday_convert(x):
    '''
    Algorithm from python-sgp4:

    from sgp4.functions import jday
    jday(x.year, x.month, x.day, x.hour, x.minute, x.second + x.microsecond * 1e-6)
    '''
    jd = (367.0 * x.year
         - 7 * (x.year + ((x.month + 9) // 12.0)) * 0.25 // 1.0
           + 275 * x.month / 9.0 // 1.0
           + x.day
         + 1721013.5)
    fr = (x.second + (x.microsecond * 1e-6) + x.minute * 60.0 + x.hour * 3600.0) / 86400.0;
    return jd, fr

def add_epoch_data(df):
    '''
    Adds the following columns to the dataset
     epoch_jd - julian datetime
     epoch_fr - julian date offset
     epoch_days - Days since 1949 Dec 31
    '''
    # Get the Julian date of the EPOCH
    df[['epoch_jd', 'epoch_fr']] = df['EPOCH'].apply(jday_convert).to_list()

    # Get the days since 1949 December 31 00:00 UT
    # This will be used when creating satobj for the test set
    # (this is needed to get the satellite position from generated TLEs
    #  because of how Satrec sgp4init() works')
    ref_date = datetime.strptime('12/31/1949 00:00:00', '%m/%d/%Y %H:%M:%S')
    df['epoch_days'] = (df['EPOCH']-ref_date)/np.timedelta64(1, 'D')

    return df

def add_satellite_position_data(df):
    # Create the satellite object (used to find satellite position)
    df['satobj'] = df.apply(lambda x: Satrec.twoline2rv(x['TLE_LINE1'], x['TLE_LINE2']), axis=1)

    # Get satellite x,y,z positions from TLE
    df['satpos'] = df.apply(lambda x: np.array(x['satobj'].sgp4(x['epoch_jd'], x['epoch_fr'])[1]), axis=1)
    return df

def create_index_map(df, debug=False, write=False, name='train', path=None,
                     compressed=False):
    '''
    This will create a map between an input record (for X_train) and a label
    record (for y_train) that will be used by the pytorch dataset class to
    dynamically build a dataset without taking up more space than is necessary.

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame containing the NORAD_CAT_ID and indexes

    debug : bool
        Print debug messages.  Default is False.

    write : bool
        Write the output to a file. Default is False.

    name : str
        name prefix of the output file.  Default is 'train'.
        Resulting filename will be something like 'train_idx_pairs.csv'

    path : str
        Path to file locations. Default is None.
        If None, it will use %GP_HIST_PATH% + '/idx_pairs'

    compressed : bool
        Compress the output file into csv.gz files.  Default is False.
        NOTE: Compression takes a lot of extra time.
    '''

    def groups(lst):
        arr = lst.copy()
        np.random.shuffle(arr)
        i=1
        if len(lst)<=1:
            return
        while True:
            if i==len(lst):
                yield tuple((arr[i-1],arr[0]))
                break
            else:
                yield tuple((arr[i-1],arr[i]))
                i+=1

    if debug:
        print('Creating index pairs.')

    # For each unique NORAD, find all TLE indexes and generate
    # a list of combinations
    idx_pairs = []
    for norad in tqdm(df['NORAD_CAT_ID'].unique()):
        norad_idxs = df[df['NORAD_CAT_ID']==norad].index.values
        if len(norad_idxs > 1):
            idx_pairs.extend(groups(norad_idxs))
    idx_pairs = np.array(idx_pairs)

    if write:
        if debug:
            print('Writing index pairs file.')

        if compressed:
            file_name = name + '_idx_pairs.csv.gz'
        else:
            file_name = name + '_idx_pairs.csv'

        if path == None:
            path = os.environ['GP_HIST_PATH'] + '/idx_pairs'

        store_path = path + '/' + file_name
        pd.DataFrame(idx_pairs,columns=['idx1','idx2']).to_csv(store_path, index=False)
        if debug:
            print(f'Index pairs wrote to {store_path}')

    if debug:
        print('Finished creating index pairs.')

    return idx_pairs

def write_data(df, name='train', path=None, compressed=False,
               sub_path='/cleaned'):
    '''
    Write the df to a "cleaned" file.

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame containing the NORAD_CAT_ID and indexes

    name : str
        name prefix of the output file.  Default is 'train'.
        Resulting filename will be something like 'train_clean.pkl'

    path : str
        Path to file locations. Default is None.
        If None, it will use %GP_HIST_PATH% + <sub_path>

    compressed : bool
        Use the file extention .pkl.gz instead of .pkl

    sub_path : str
        sub path where to write pickle files to.  Default is '/cleaned'.
        Must be prefixed with '/'
        Not needed if path specified.
    '''
    if compressed:
        file_name = name + '_clean.pkl.gz'
    else:
        file_name = name + '_clean.pkl'

    if path == None:
        path = os.environ['GP_HIST_PATH'] + sub_path

    store_path = path + '/' + file_name
    df.to_pickle(store_path)

def load_index_map(name='train', path=None, compressed=False):
    '''
    Reads a file containing the index_pair list to a numpy array

    Parameters
    ----------
    name : str
        name prefix of the output file.  Default is 'train'.

    path : str
        Path to file locations. Default is None.

    compressed : bool
        Use the file extention .csv.gz instead of .csv
    '''
    if compressed:
        file_name = name + '_idx_pairs.csv.gz'
    else:
        file_name = name + '_idx_pairs.csv'

    if path == None:
        path = os.environ['GP_HIST_PATH'] + '/idx_pairs'

    store_path = path + '/' + file_name

    idx_pairs = pd.read_csv(store_path).to_numpy()
    return idx_pairs

if __name__ == '__main__':
    import argparse
    from time import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_idx_map', action='store_const', const=True, default=False, help='Load index map')

    args = parser.parse_args()

    if args.load_idx_map:
        ts = time()
        load_index_map()
        print(f'  Took {round(time()-ts)} seconds')
