import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def load_norads(data_types=['train'], debug=False):
    '''
    Loads the NORAD IDs for each specific phase

    Parameters
    ----------
    type : list
        list of strings containing the following possibilities: train, validate, test (default is ['train'])

        NOTES:
        validate and test are the same data.
        To access the final test data, use 'secret_test'

    Returns
    -------
    dict
        a dictionary of lists containing the NORAD IDs for each data_type

    Raises
    ------
    ValueError
        If unexpected data_type
    '''

    norad_lists = {}

    for data_type in data_types:
        if data_type == 'train':
            file_name = 'train_norads.pkl.gz'
        elif data_type in ('validate', 'test'):
            file_name = 'validate_norads.pkl.gz'
            data_type = 'test'
        elif data_type == 'secret_test':
            file_name = 'test_norads.pkl.gz'
        else:
            raise ValueError(f'Unexpected data_type when loading NORAD IDs: {data_type}')

        if debug:
            print(f'Loading NORAD list from file: {file_name}')

        norad_df = pd.read_pickle(os.environ['my_home_path'] + '/data/split_by_norad/' + file_name)
        norad_list = norad_df.norad.to_list()
        norad_lists[data_type] = norad_list

    # Show the results of the norad id load
    if debug:
        for k,v in norad_lists.items():
            print(f' {k} has {len(v)} NORAD IDs')

    return norad_lists

def load_data(norad_lists, use_all_data=False, debug=False):
    '''
    Load gp_history csv.gz files into a pandas dataframe
    '''

    if use_all_data==True:
        csv_store_path = os.environ['GP_HIST_PATH']
    else:
        csv_store_path = os.environ['my_home_path'] + '/data/space-track-gp-hist-sample'

    if debug:
        print(f'Loading files from path: {csv_store_path}')

    necessary_columns = ['NORAD_CAT_ID','OBJECT_TYPE','OBJECT_NAME','TLE_LINE1','TLE_LINE2',
                         'MEAN_MOTION_DOT', 'MEAN_MOTION_DDOT', 'BSTAR', 'INCLINATION', 'RA_OF_ASC_NODE',
                         'ECCENTRICITY', 'ARG_OF_PERICENTER', 'MEAN_ANOMALY', 'MEAN_MOTION', 'EPOCH']

    files = sorted([x for x in os.listdir(f'{csv_store_path}/') if x.endswith(".csv.gz")])
    df_list = []
    for f in tqdm(files):
        df = pd.read_csv(f'{csv_store_path}/{f}',
                         parse_dates=['EPOCH'],
                         infer_datetime_format=True,
                         compression='gzip',
                         low_memory=False)
        df = df[df.NORAD_CAT_ID.isin(norad_list)][necessary_columns]
        df_list.append(df)

    dfs = pd.concat(df_list)
    dfs = dfs.reset_index()
    return dfs

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', help='While data to load (train/validate/test). NOTE: validate and test are the same.  To access final test, use secret_test')
    parser.add_argument('--use_all_data', action='store_const', const=True, default=False, help='Use all gp_history data.')

    args = parser.parse_args()

    # Get the NORAD list
    #norad_list = load_norads(args.data_type, True)
    norad_list = load_norads(['train', 'test', 'train', 'validate'], True)

    # Get the pandas dataframe
    #df = load_data(norad_list, args.use_all_data, True)

    #print(df.head())
