import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def load_norads(data_types=['train'], debug=False):
    '''
    Loads the NORAD IDs for each specific phase

    Parameters
    ----------
    data_types : list
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
        If unexpected data_types
    '''

    norad_lists = {}

    if len(data_types) == 0:
        raise ValueError('data_types must contain at least one item')

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

    Parameters
    ----------
    norad_lists : dict
        key : data_type (such as train, validate, or test )
        value : list of ints containing the NORAD IDs within in output dataframe

    Returns
    -------
    dict
        a dictionary of pandas dataframes containing the TLE gp_history for all
        NORAD IDs for each data_type
    '''

    if len(norad_lists.keys()) == 0:
        raise ValueError('norad_lists must contain at least one list')

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
    df_dict = {}
    for data_type in norad_lists.keys():
        df_dict[data_type] = []

    for f in tqdm(files):
        df = pd.read_csv(f'{csv_store_path}/{f}',
                         parse_dates=['EPOCH'],
                         infer_datetime_format=True,
                         compression='gzip',
                         low_memory=False)
        for data_type, norad_list in norad_lists.items():
            odf = df[df.NORAD_CAT_ID.isin(norad_list)][necessary_columns]
            df_dict[data_type].append(odf)

    df_out = {}
    for data_type, df_list in df_dict.items():
        df_out[data_type] = pd.concat(df_list).reset_index()
    return df_out

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_const', const=True, default=False, help='Load train data')
    parser.add_argument('--validate', action='store_const', const=True, default=False, help='Load validate data')
    parser.add_argument('--secret', action='store_const', const=True, default=False, help='Load secret test data')
    parser.add_argument('--use_all_data', action='store_const', const=True, default=False, help='Use all gp_history data.')

    args = parser.parse_args()

    # Get the NORAD list
    data_types = []
    if args.train:
        data_types.append('train')
    if args.validate:
        data_types.append('validate')
    if args.secret:
        data_types.append('secret_test')
    norad_lists = load_norads(data_types, True)

    # Get the pandas dataframe
    df_dict = load_data(norad_lists, args.use_all_data, True)

    for k,v in df_dict.items():
        print(f'{k} has {len(v)} items:')
        print(v.head())
