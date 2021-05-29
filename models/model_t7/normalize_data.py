import pandas as pd
import numpy as np
from sgp4.api import Satrec, WGS72
from datetime import datetime
from tqdm import tqdm_notebook as tqdm
import os
import concurrent.futures

def normalize_all_columns(df, reverse=False):
    '''
    Normalizes all dataframe columns

    Parameters
    ----------
    df : DataFrame or Series
        gp_history dataframe to be normalized

    reverse : bool
        Reverse normalization

    Returns
    -------
    Dataframe or Series
        The normalized result
    '''
#     'X_INCLINATION_1', 'y_INCLINATION'
    from_180_deg = ['X_INCLINATION_1']
#     'X_MEAN_ANOMALY_1', 'X_RA_OF_ASC_NODE_1', 'X_ARG_OF_PERICENTER_1', 'y_RA_OF_ASC_NODE', 'y_ARG_OF_PERICENTER'
    from_360_deg = ['X_MEAN_ANOMALY_1', 'X_RA_OF_ASC_NODE_1', 'X_ARG_OF_PERICENTER_1']

    df[from_180_deg] = normalize(df[from_180_deg],min=0,max=180,reverse=reverse)
    df[from_360_deg] = normalize(df[from_360_deg],min=0,max=360,reverse=reverse)
    
#     'X_delta_EPOCH'
    df['X_delta_EPOCH'] = normalize(df['X_delta_EPOCH'],min=0,max=7,reverse=reverse)
    
#     'X_EPOCH_JD_1', 'X_EPOCH_JD_2', 
    df['X_EPOCH_JD_1'] = normalize(df['X_EPOCH_JD_1'], min=2447892.5, max=2459305.5,reverse=reverse)
    df['X_EPOCH_JD_2'] = normalize(df['X_EPOCH_JD_2'], min=2447892.5, max=2459305.5,reverse=reverse)
    
#     'X_ECCENTRICITY_1', 'y_ECCENTRICITY'
    df['X_ECCENTRICITY_1'] = normalize(df['X_ECCENTRICITY_1'],min=0,max=0.25,reverse=reverse)

#     'X_SUNSPOTS_1D_1', 'X_SUNSPOTS_3D_1', 'X_SUNSPOTS_7D_1'
    df['X_SUNSPOTS_1D_1'] = normalize(df['X_SUNSPOTS_1D_1'],min=0,max=500,reverse=reverse)
    df['X_SUNSPOTS_3D_1'] = normalize(df['X_SUNSPOTS_3D_1'],min=0,max=500,reverse=reverse)
    df['X_SUNSPOTS_7D_1'] = normalize(df['X_SUNSPOTS_7D_1'],min=0,max=500,reverse=reverse)
    
#     'X_MEAN_MOTION_1', 'y_MEAN_MOTION'
    df['X_MEAN_MOTION_1'] = normalize(df['X_MEAN_MOTION_1'],min=11.25,max=20,reverse=reverse)

#     'X_YEAR_1'
    df['X_YEAR_1'] = normalize(df['X_YEAR_1'],min=1990,max=2021,reverse=reverse)

#     'X_SAT_RX_1', 'X_SAT_RY_1', 'X_SAT_RZ_1'
    sat_r = ['y_SAT_RX', 'y_SAT_RY', 'y_SAT_RZ', 'X_SAT_RX_1', 'X_SAT_RY_1', 'X_SAT_RZ_1', 'X_SGP4_SAT_RX', 'X_SGP4_SAT_RY', 'X_SGP4_SAT_RZ']
    df[sat_r] = normalize(df[sat_r], min=-8000, max=8000, range=[-1,1],reverse=reverse)
    
#     'X_SEMIMAJOR_AXIS_1'
    df['X_SEMIMAJOR_AXIS_1'] = normalize(df['X_SEMIMAJOR_AXIS_1'], min=6500, max=8500, reverse=reverse)

#     'X_PERIOD_1'
    df['X_PERIOD_1'] = normalize(df['X_PERIOD_1'], min=8.5, max=13, reverse=reverse)

#     'X_APOAPSIS_1', 'X_PERIAPSIS_1'
    apo_peri = ['X_APOAPSIS_1', 'X_PERIAPSIS_1']
    df[apo_peri] = normalize(df[apo_peri], min=100, max=4000, reverse=reverse)
    
#     'X_RCS_SIZE_1'
    df['X_RCS_SIZE_1'] = normalize(df['X_RCS_SIZE_1'], min=-1, max=2, reverse=reverse)

#     'X_SAT_VX_1', 'X_SAT_VY_1', 'X_SAT_VZ_1'
    sat_v = ['y_SAT_VX', 'y_SAT_VY', 'y_SAT_VZ', 'X_SAT_VX_1', 'X_SAT_VY_1', 'X_SAT_VZ_1', 'X_SGP4_SAT_VX', 'X_SGP4_SAT_VY', 'X_SGP4_SAT_VZ']
    df[sat_v] = normalize(df[sat_v], min=-8, max=8, range=[-1,1],reverse=reverse)
    
    # not normalized
# 'X_EPOCH_FR_1', 'X_EPOCH_FR_2' # already in range
# 'X_BSTAR_1', 'y_BSTAR' # acceptable range
# 'X_MEAN_MOTION_DOT_1'  # acceptable range
# 'X_MEAN_ANOMALY_COS_1',
# 'X_MEAN_ANOMALY_SIN_1',
# 'X_INCLINATION_COS_1',
# 'X_INCLINATION_SIN_1',
# 'X_RA_OF_ASC_NODE_COS_1',
# 'X_RA_OF_ASC_NODE_SIN_1',
# 'X_DAY_OF_YEAR_COS_1',
# 'X_DAY_OF_YEAR_SIN_1',
# 'X_AIR_MONTH_AVG_TEMP_1',
# 'X_WATER_MONTH_AVG_TEMP_1',

    return df

def normalize(df,max=None,min=None,mean=None,std=None,range=[0,1],reverse=False):
    '''
    Normalizes dataframe columns

    Parameters
    ----------
    df : DataFrame or Series
        All columns to be normalized using either min-max or around the mean
        normalization.  Only one method can be applied.

    min / max : float
        Minimum and Maximum values for min-max normalization

    range : list
        The lower and upper bounds of min-max normalization

    mean / std : float
        Mean and standard deviation for normalizing around the mean

    reverse : bool
        Reverse normalization

    Returns
    -------
    Dataframe or Series
        The normalized result
    '''
    if not reverse:
        if mean!=None and std!=None:
            return (df - mean)/std
        elif min!=None and max!=None:
            if range == [0,1]:
                return (df - min) / (max - min)
            else:
                b = range[1]
                a = range[0]
                return (b-a) * (df - min) / (max - min) + a
        else:
            raise ValueError(f"Normalization type is not recognized. Require max/min or mean/std.")
    else:
        if mean!=None and std!=None:
            return (df * std) + mean
        elif min!=None and max!=None:
            if range == [0,1]:
                return (df * (max - min)) + min
            else:
                b = range[1]
                a = range[0]
                return ((df-a)/(b-a) * (max - min)) + min
        else:
            raise ValueError(f"Normalization type is not recognized. Require max/min or mean/std.")