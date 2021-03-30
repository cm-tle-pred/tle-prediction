# Get NORAD Counts
This module will look across the entire gp_history and extract the NORAD IDs that are DEBRIS.  The reason is because we don't want to include satellites that maneuver within our model so we will focus on DEBRIS with the assumption these don't maneuver.

NOTE: this module uses parallel processing.

## Usage
!python multiproc_norad_counts.py

## Output
/data/norad_debris_count.csv

## Execution time
Nick's local machine: 180s

Tim's local machine: 15min
