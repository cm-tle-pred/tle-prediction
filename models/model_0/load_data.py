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
    Concurrent/Multiprocessing task that loads a csv.gz file
    '''

    necessary_columns = ['NORAD_CAT_ID','OBJECT_TYPE','OBJECT_NAME','TLE_LINE1','TLE_LINE2',
                         'MEAN_MOTION_DOT', 'MEAN_MOTION_DDOT', 'BSTAR', 'INCLINATION', 'RA_OF_ASC_NODE',
                         'ECCENTRICITY', 'ARG_OF_PERICENTER', 'MEAN_ANOMALY', 'MEAN_MOTION', 'EPOCH']
    df_dict = {}
    for data_type in norad_lists.keys():
        df_dict[data_type] = []
    try:
        # df = pd.read_csv(file_path, compression='gzip', low_memory=False)
        # df = df[(df.MEAN_MOTION > 11.25) & (df.ECCENTRICITY < 0.25) & (df.OBJECT_TYPE != 'PAYLOAD')]
        df = pd.read_csv(file_path,
                         parse_dates=['EPOCH'],
                         infer_datetime_format=True,
                         compression='gzip',
                         low_memory=False)
        for data_type, norad_list in norad_lists.items():
            df_dict[data_type] = df[df.NORAD_CAT_ID.isin(norad_list)][necessary_columns]
    except:
        raise Exception(f'Failed to open {file_path}')
    return df_dict

def load_data_multi(norad_lists, use_all_data=False, debug=False, threaded=False):
    '''
    Load gp_history csv.gz files into a pandas dataframe using multiple
    processes or threads

    Parameters
    ----------
    norad_lists : dict
        key : data_type (such as train, validate, or test )
        value : list of ints containing the NORAD IDs within in output dataframe

    use_all_data : bool
        Use all the .csv.gz gp_history files.  Default is False.

    debug : bool
        Print debug messages.  Default is False.

    threaded : bool
        Use threads instead of processes.  Default is False.

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

    files = sorted([x for x in os.listdir(f'{csv_store_path}/') if x.endswith(".csv.gz")])
    df_dict = {}
    for data_type in norad_lists.keys():
        df_dict[data_type] = []

    if threaded:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(__load_task, files)
            if debug:
                print('Finished loading all files. Starting assembly.')
            for result in results:
                for data_type, df in result:
                    df_dict[data_type].append(df)
            if debug:
                print('Finished assembly.')

    else:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(__load_task, files)
            if debug:
                print('Finished loading all files. Starting assembly.')
            for result in results:
                for data_type, df in result:
                    df_dict[data_type].append(df)
            if debug:
                print('Finished assembly.')

    df_out = {}
    for data_type, df_list in df_dict.items():
        df_out[data_type] = pd.concat(df_list).reset_index()
    return df_out

def load_data(norad_lists, use_all_data=False, debug=False):
    '''
    Load gp_history csv.gz files into a pandas dataframe

    Parameters
    ----------
    norad_lists : dict
        key : data_type (such as train, validate, or test )
        value : list of ints containing the NORAD IDs within in output dataframe

    use_all_data : bool
        Use all the .csv.gz gp_history files.  Default is False.

    debug : bool
        Print debug messages.  Default is False.

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

def write_raw_data(df_dict, use_all_data=False, debug=False, sub_path='/raw_compiled'):
    '''
    Writes all dataframes in df_dict to separate pickle files.

    Parameters
    ----------
    df_dict : dict
        key : data_type (such as train, validate, or test )
        value : list of ints containing the NORAD IDs within in output dataframe

    use_all_data : bool
        Use all the .csv.gz gp_history files.  Default is False.

    debug : bool
        Print debug messages.  Default is False.

    sub_path : str
        sub path where to write pickle files to.  Default is raw_compiled.

    Raises
    ------
    ValueError
        If df_dict is empty
    '''

    #if len(df_dict.keys()) == 0 or all([type(x)==type(pd.DataFrame()) for x in df_dict.values()]):
    if len(df_dict.keys()) == 0:
        raise ValueError('df_dict must contain at least item and all must be pandas dataframes')

    if use_all_data==True:
        store_path = os.environ['GP_HIST_PATH'] + sub_path
    else:
        store_path = os.environ['my_home_path'] + '/data/space-track-gp-hist-sample' + sub_path

    for data_type, df in df_dict.items():
        file_name = data_type  + '.pkl.gz'
        path = store_path + '/' + file_name
        if debug:
            print(f'Writing raw data for {data_type} to: {path}')
        df.to_pickle(path)
    if debug:
        print(f'Done writing output.')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_const', const=True, default=False, help='Load train data')
    parser.add_argument('--validate', action='store_const', const=True, default=False, help='Load validate data')
    parser.add_argument('--secret', action='store_const', const=True, default=False, help='Load secret test data')
    parser.add_argument('--use_all_data', action='store_const', const=True, default=False, help='Use all gp_history data.')
    parser.add_argument('--write', action='store_const', const=True, default=False, help='Write output to pickle files.')
    parser.add_argument('--multiprocess', action='store_const', const=True, default=False, help='Use multiprocessing for loading.')
    parser.add_argument('--multithreaded', action='store_const', const=True, default=False, help='Use multithreading for loading.')

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
    if args.multiprocess:
        df_dict = load_data_multi(norad_lists, args.use_all_data, True)
    elif args.multithreaded:
        df_dict = load_data_multi(norad_lists, args.use_all_data, True, True)
    else:
        df_dict = load_data(norad_lists, args.use_all_data, True)

    for k,v in df_dict.items():
        print(f'{k} has {len(v)} items:')
        print(v.head())

    # Write output
    if args.write:
        write_raw_data(df_dict, args.use_all_data, True)
