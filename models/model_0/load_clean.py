import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def load_norads(data_type='train'):
    '''
    Loads the NORAD IDs for each specific phase

    Parameters
    ----------
    type : str
        one of the following: train, validate, test (default is train)

        NOTES:
        validate and test are the same data.
        To access the final test data, use 'secret_test'

    Returns
    -------
    list
        a list of NORAD IDs

    Raises
    ------
    ValueError
        If unexpected data_type
    '''
    if data_type == 'train':
        file_name = 'train_norads.pkl.gz'
    elif data_type in ('validate', 'test'):
        file_name = 'validate_norads.pkl.gz'
    elif data_type == 'secret_test':
        file_name = 'test_norads.pkl.gz'
    else:
        raise ValueError(f'Unexpected data_type when loading NORAD IDs: {data_type}')

    norad_df = pd.read_pickle(os.environ['my_home_path'] + '/data/split_by_norad/' + file_name)
    norad_list = norad_df.norad.to_list()
    return norad_list

def load_data(norad_list, use_all_data=False):
    '''
    Load gp_history csv.gz files into a pandas dataframe
    '''
    if use_all_data==0:
        csv_store_path = os.environ['GP_HIST_PATH']
    else:
        csv_store_path = os.environ['my_home_path'] + '/data/space-track-gp-hist-sample'

    necessary_columns = ['NORAD_CAT_ID','OBJECT_TYPE','OBJECT_NAME','TLE_LINE1','TLE_LINE2',
                         'MEAN_MOTION_DOT', 'MEAN_MOTION_DDOT', 'BSTAR', 'INCLINATION', 'RA_OF_ASC_NODE',
                         'ECCENTRICITY', 'ARG_OF_PERICENTER', 'MEAN_ANOMALY', 'MEAN_MOTION', 'EPOCH']
    dfs = None
    files = sorted([x for x in os.listdir(f'{csv_store_path}/') if x.endswith(".csv.gz")])
    for f in tqdm(files):
        df = pd.read_csv(f'{csv_store_path}/{f}', parse_dates=['EPOCH'], infer_datetime_format=True, compression='gzip')
        df = df[df.NORAD_CAT_ID.isin(norad_list)][necessary_columns]

        # Since animated gabbard diagrams are generated per frame, we can revert the scaling when we plot the graphs
        if dfs is None:
            dfs = df
        elif len(df) > 0:
            dfs = pd.concat([dfs,df])

    dfs = dfs.reset_index()
    return dfs

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='While data to load (train/validate/test) Default=train.  NOTE: validate and test are the same.  To access final test, use secret_test')
    parser.add_argument('use_all_data', help='Use all gp_history data (1=Yes/0=No) Default=0')


    # TO DO - make the use_all_data not use a 1/0 value.  see here: https://docs.python.org/3/library/argparse.html

    # parser.add_argument(
    #     'input_file', help='the raw trips without temp file (CSV)')
    # parser.add_argument(
    #     'output_file', help='the clean trips without temp file (CSV)')
    args = parser.parse_args()


    # clean = clean(args.input_file)
    # clean.to_csv(args.output_file, index=False)
