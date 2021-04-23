## Load Raw Data into Train/Val/Test sets and Save

Takes the norad split list from `0_preproc` step and saves train/val/holdout dataframes as pkl files.

Loads up the data from CSV raw and saved in the shared data directory at `/mistorage/mads/data`

2 versions saved

* `0_full` includes the 2 TLE Lines and OBJECT_NAME
* `0_min` excludes those 3 fields