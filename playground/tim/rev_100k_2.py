import os
import concurrent.futures
import pandas as pd
from collections import Counter
from time import time
from sys import platform
from tqdm import tqdm

def main():
    csv_store_path = os.environ['GP_HIST_PATH']
    results = []
    ts = time()
    
    idf = pd.read_pickle("./ignore/rev100k.pkl").sort_values(by=["EPOCH"])
    idf = idf[idf.REV_AT_EPOCH==10000]
    ids = idf.NORAD_CAT_ID.unique()
    
    for p in tqdm(sorted([x for x in os.listdir(f'{csv_store_path}/') if x.endswith(".csv.gz")])):
        df = pd.read_csv(f"{csv_store_path}/{p}", compression='gzip', low_memory=False)
        df = df[df.NORAD_CAT_ID.isin(ids)]
        results.append(df)
    final_df = pd.concat(results)
    print(f'Took {time()-ts}')
    final_df.to_pickle(f"./ignore/rev100k_2.pkl")


if __name__ == '__main__':
    main()