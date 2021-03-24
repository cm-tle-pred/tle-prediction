import concurrent.futures
import pandas as pd
from collections import Counter
from time import time

def concurrent_task(file_path):
    try:
        df = pd.read_csv(file_path, compression='gzip', low_memory=False)
        df = df[(df.MEAN_MOTION > 11.25) & (df.ECCENTRICITY < 0.25)]
    except:
        raise Exception(f'Failed to open {file_path}')
    return Counter(df.NORAD_CAT_ID.to_list())

def write_output(all_counts):
    with open('multiproc_output.txt', 'w+') as f:
        for k,v in all_counts.items():
            f.write(str(k) + ',' + str(v) + '\n')

def main():
    file_list = 'all_files.txt'
    
    ts = time()
    all_counts = Counter()
    files = [file[:-1] for file in open(file_list).readlines()]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(concurrent_task, files)
        for result in results:
            all_counts += result
    print(f'Took {time()-ts}')
    write_output(all_counts)
    

if __name__ == '__main__':
    main()
