from datetime import datetime

import pandas as pd
from datetime import datetime
event_file = r"./dataset/lastfm/processed_events_shuffled.dat"
# data = pd.read_csv(event_file, sep='\t', header=0, names=['user', 'timestamp', 'armpool'])
# count=0
# for i, line in data.iterrows():
#     count+=1
#     # if count == 1:
#     #     continue
#     x = line["timestamp"]
#     # if count == 1:
#     #     continue
#     print(count)
#     datetime.utcfromtimestamp(int(x)/1000)

x = "-1"
print(int(x)) 
print(datetime.utcfromtimestamp(-420771600000/1000.0) )