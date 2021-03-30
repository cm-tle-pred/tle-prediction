import pandas as pd
import numpy as np
from sgp4.api import Satrec, WGS72
from datetime import datetime

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

if __name__ == '__main__':
    import argparse
    from time import time
