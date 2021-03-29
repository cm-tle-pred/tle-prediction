import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import concurrent.futures
from itertools import repeat

def load_norads(data_types=['train'], debug=False):
    '''
    Loads the NORAD IDs for each data_type

    Parameters
    ----------
    data_types : list
        list of strings containing the following possibilities: train, validate,
        test.  Default is ['train'].

        NOTES:
        validate and test are the same data.
        To access the final test data, use 'secret_test'
        Expected filenames
         - train: train_norads.pkl.gz
         - validate / test: validate_norads.pkl.gz
         - secret_test: test_norads.pkl.gz

    debug : bool
        Print debug messages.  Default is False.

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

def __load_task(norad_lists, file_path):
    '''
    Concurrent/Multiprocessing supported task that loads a csv.gz file
    '''
    necessary_columns = ['NORAD_CAT_ID','OBJECT_TYPE','OBJECT_NAME','TLE_LINE1',
                         'TLE_LINE2','MEAN_MOTION_DOT', 'MEAN_MOTION_DDOT',
                         'BSTAR', 'INCLINATION', 'RA_OF_ASC_NODE',
                         'ECCENTRICITY', 'ARG_OF_PERICENTER', 'MEAN_ANOMALY',
                         'MEAN_MOTION', 'EPOCH']
    df_dict = {}
    for data_type in norad_lists.keys():
        df_dict[data_type] = []
    try:
        df = pd.read_csv(file_path,
                         parse_dates=['EPOCH'],
                         infer_datetime_format=True,
                         compression='gzip',
                         low_memory=False)
        for data_type, norad_list in norad_lists.items():
            df_dict[data_type] = df[df.NORAD_CAT_ID.isin(norad_list)][necessary_columns]
    except Exception as e:
        raise Exception(f'Failed to open {file_path}.  Error: {e}')
    return df_dict

def load_data(norad_lists, use_all_data=False, debug=False, threaded=False,
              multiproc=False):
    '''
    Load gp_history csv.gz files into a pandas dataframe.

    Parameters
    ----------
    norad_lists : dict
        key : data_type (such as train, validate, or test )
        value : list of ints containing the NORAD IDs within in output dataframe

    use_all_data : bool
        Use all the .csv.gz gp_history files.  Default is False.
        False: %GP_HIST_PATH%
        True: %my_home_path%/data/space-track-gp-hist-sample

    debug : bool
        Print debug messages.  Default is False.

    threaded : bool
        Use multiple threads.  Default is False.

    multiproc : bool
        Use multiple processes.  Default is False.

    Returns
    -------
    dict
        a dictionary of pandas dataframes containing the TLE gp_history for all
        NORAD IDs for each data_type

    Raises
    ------
    ValueError
        If norad_lists is empty
    '''

    if len(norad_lists.keys()) == 0:
        raise ValueError('norad_lists must contain at least one list')

    if use_all_data==True:
        csv_store_path = os.environ['GP_HIST_PATH']
    else:
        csv_store_path = os.environ['my_home_path'] + '/data/space-track-gp-hist-sample'

    if debug:
        print(f'Loading files from path: {csv_store_path}')

    files = sorted([f'{csv_store_path}/{x}' for x in os.listdir(f'{csv_store_path}/') if x.endswith(".csv.gz")])
    df_dict = {}
    for data_type in norad_lists.keys():
        df_dict[data_type] = []

    if threaded:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(__load_task, repeat(norad_lists), files), total=len(files)))
            for result in results:
                for data_type, df in result.items():
                    df_dict[data_type].append(df)
            if debug:
                print('Finished loading.')
    elif multiproc:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(__load_task, repeat(norad_lists), files), total=len(files)))
            for result in results:
                for data_type, df in result.items():
                    df_dict[data_type].append(df)
            if debug:
                print('Finished loading.')
    else:
        for f in tqdm(files):
            for data_type, df in __load_task(norad_lists, f).items():
                df_dict[data_type].append(df)
        print('Finished loading.')

    df_out = {}
    for data_type, df_list in tqdm(df_dict.items()):
        df_out[data_type] = pd.concat(df_list).reset_index(drop=True)
    if debug:
        print('Finished assembling.')
    return df_out

def __write_task(df, path, debug=False):
    '''
    Concurrent/Multiprocessing supported task that saves a df to pickle file
    '''
    if debug:
        print(f'Writing raw data for to: {path}')
    df.to_pickle(path)
    return path

def write_data(df_dict, use_all_data=False, debug=False,
                   sub_path='/raw_compiled', threaded=False,
                   multiproc=False):
    '''
    Writes all dataframes in df_dict to separate pickle files.

    Parameters
    ----------
    df_dict : dict
        key : data_type (such as train, validate, or test )
        value : list of ints containing the NORAD IDs within in output dataframe

    use_all_data : bool
        Use the path with all the .csv.gz gp_history files.  Default is False.
        False: %GP_HIST_PATH% + <sub_path>
        True: %my_home_path%/data/space-track-gp-hist-sample + <sub_path>

    debug : bool
        Print debug messages.  Default is False.

    sub_path : str
        sub path where to write pickle files to.  Default is '/raw_compiled'.
        Must be prefixed with '/'

    threaded : bool
        Use multiple threads.  Default is False.

    multiproc : bool
        Use multiple processes.  Default is False.
        NOTE: Not used.  Will instead enable multithrading

    Raises
    ------
    ValueError
        If df_dict is empty
    '''

    if len(df_dict.keys()) == 0:
        raise ValueError('df_dict must contain at least one item and all must be pandas dataframes')

    if use_all_data==True:
        store_path = os.environ['GP_HIST_PATH'] + sub_path
    else:
        store_path = os.environ['my_home_path'] + '/data/space-track-gp-hist-sample' + sub_path
    paths = [store_path + '/' + data_type + '.pkl.gz' for data_type in df_dict.keys()]

    if debug:
        print(f'Saving files to path: {store_path}')

    if multiproc:
        # Force threaded to save memory
        threaded=True

    if threaded:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            tasks = [executor.submit(__write_task, df, paths[i], debug) for i, df in enumerate(df_dict.values())]
            for t in concurrent.futures.as_completed(tasks):
                print(f'Finished saving {t.result()}')
    else:
        for i, df in enumerate(df_dict.values()):
            path = paths[i]
            __write_task(df, path, debug)
        if debug:
            print(f'Done writing output.')

if __name__ == '__main__':
    import argparse
    from time import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_const', const=True, default=False, help='Load train data')
    parser.add_argument('--validate', action='store_const', const=True, default=False, help='Load validate data')
    parser.add_argument('--secret', action='store_const', const=True, default=False, help='Load secret test data')
    parser.add_argument('--use_all_data', action='store_const', const=True, default=False, help='Use all gp_history data.')
    parser.add_argument('--write', action='store_const', const=True, default=False, help='Write output to pickle files.')
    parser.add_argument('--multiprocess', action='store_const', const=True, default=False, help='Use multiprocessing for loading.')
    parser.add_argument('--multithreaded', action='store_const', const=True, default=False, help='Use multithreading for loading.')

    args = parser.parse_args()

    # Load the NORAD list
    data_types = []
    if args.train:
        data_types.append('train')
    if args.validate:
        data_types.append('validate')
    if args.secret:
        data_types.append('secret_test')
    norad_lists = load_norads(data_types, debug=True)

    # Load the pandas dataframe
    ts = time()
    if args.multiprocess:
        df_dict = load_data(norad_lists, use_all_data=args.use_all_data, debug=True, multiproc=True)
    elif args.multithreaded:
        df_dict = load_data(norad_lists, use_all_data=args.use_all_data, debug=True, threaded=True)
    else:
        df_dict = load_data(norad_lists, use_all_data=args.use_all_data, debug=True)
    print(f'Loading and assembling took {round(time()-ts)} seconds')

    # Write output
    if args.write:
        ts = time()
        if args.multiprocess:
            write_data(df_dict, use_all_data=args.use_all_data, debug=True, multiproc=True)
        elif args.multithreaded:
            write_data(df_dict, use_all_data=args.use_all_data, debug=True, threaded=True)
        else:
            write_data(df_dict, use_all_data=args.use_all_data, debug=True)
        print(f'Saving took {round(time()-ts)} seconds')
