import os
import concurrent.futures
import pandas as pd
from collections import Counter
from time import time

def build_file_list(file_name):
    '''
    Create a new file containing a list of all the
    file names to parse.
    '''
    csv_store_path = os.environ['GP_HIST_PATH']
    files = sorted([x for x in os.listdir(f'{csv_store_path}/') if x.endswith(".csv.gz")])
    with open(file_name, 'w+') as f:
        for file in files:
            f.write(csv_store_path + '\\' + file + '\n')

def task(file_path):
    '''
    Multiprocessing Task that gets the count of records for each
    NORAD ID in a single file.
    '''
    try:
        df = pd.read_csv(file_path, compression='gzip', low_memory=False)
        df = df[(df.MEAN_MOTION > 11.25) & (df.ECCENTRICITY < 0.25) & (df.OBJECT_TYPE != 'PAYLOAD')]
    except:
        raise Exception(f'Failed to open {file_path}')
    return Counter(df.NORAD_CAT_ID.to_list())

def write_output(all_counts):
    '''
    Writes the dictionary containing all the NORAD IDs
    to a file.
    '''
    with open('multiproc_output.txt', 'w+') as f:
        for k,v in all_counts.items():
            f.write(str(k) + ',' + str(v) + '\n')

def main():
    file_list = 'all_files.txt'
    build_file_list(file_list)

    ts = time()
    all_counts = Counter()
    files = [file[:-1] for file in open(file_list).readlines()]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(task, files)
        for result in results:
            all_counts += result

    print(f'Took {time()-ts}')
    write_output(all_counts)

if __name__ == '__main__':
    main()
