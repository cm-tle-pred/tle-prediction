import os
import concurrent.futures
import pandas as pd
from collections import Counter
from time import time
from sys import platform
from tqdm import tqdm

def build_file_list(file_name):
    '''
    Create a new file containing a list of all the
    file names to parse.
    '''
    csv_store_path = os.environ['GP_HIST_PATH']
    files = sorted([x for x in os.listdir(f'{csv_store_path}/') if x.endswith(".csv.gz")])
    with open(file_name, 'w+') as f:
        for file in files:
            f.write(csv_store_path + os.path.sep + file + '\n')

def task(file_path):
    '''
    Multiprocessing Task that gets the count of records for each
    NORAD ID in a single file.
    '''
    try:
        df = pd.read_csv(file_path, compression='gzip', low_memory=False)
        df = df[(df.MEAN_MOTION > 11.25) & (df.ECCENTRICITY < 0.25) & (df.OBJECT_TYPE != 'PAYLOAD')]
        
        norad_ids = [25988, 26285, 12223, 16720]
        df = df[df.NORAD_CAT_ID.isin(norad_ids)]
    except:
        raise Exception(f'Failed to open {file_path}')
    return df


def main():
    file_list = 'all_files.txt'
    build_file_list(file_list)

    ts = time()
    final_df = None
    files = [file[:-1] for file in open(file_list).readlines()]
    results = None
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(task, files), total=len(files)))

    final_df = pd.concat(results)
    print(f'Took {time()-ts}')
    final_df.to_pickle(f"./sample.pkl")


if __name__ == '__main__':
    main()