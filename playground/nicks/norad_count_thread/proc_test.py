import os
import pandas as pd
pd.options.display.html.table_schema = True
pd.options.display.max_rows = None

data_path = os.environ['my_home_path'] + '\data\space-track-gp-hist-sample'

files = sorted([x for x in os.listdir(f'{data_path}\\') if x.endswith(".csv.gz")])

df = pd.read_csv(data_path + '\\' + files[1])
df.head(20)
