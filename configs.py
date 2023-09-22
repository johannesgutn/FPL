import pandas as pd
pd.options.mode.chained_assignment = None
import json
import requests
import os
try:
    from pandas import json_normalize
except ImportError:
    from pandas.io.json import json_normalize
import numpy as np
from pathlib import Path

from fpl import FPL
import aiohttp
import asyncio


def get_latest_GW(json_data):
    # Making dataframe and saving newest raw_data
    df = json_normalize(json_data['events'])
    last_gw=0
    for index, row in df.iterrows():
        if row['finished']:
            last_gw = row['id']
    
    return last_gw

def get_season():
    with open('json/fpl_events.json') as json_data:
        d = json.load(json_data)
    df=json_normalize(d['events'])
    start_year=int(df.loc[1,'deadline_time'][2:4])
    end_year=start_year+1
    season=f'{start_year}-{end_year}'
    return season

def get_json(file_path, url):
    r = requests.get(url)
    jsonResponse = r.json()
    with open(file_path, 'w') as outfile:
        json.dump(jsonResponse, outfile)



# pd.merge() probably does this, so check it out
def insert(df_to, col_to, col_match_to, df_from, col_from, col_match_from):
    df_to[col_to]=df_to[col_match_to]
    di = df_from.set_index(col_match_from)[col_from].to_dict()
    df_to[col_to] = df_to[col_to].map(di)
    
    return df_to

# Not an ideal structure to have some random variables in the middle of the list of functions
# Making global variables
user = os.getlogin()
wdir=f'/Users/johannes/Library/CloudStorage/Dropbox/ml/FPL/'
season = get_season()
last_season = f'{int(season[0:2])-1}-{int(season[0:2])}'

# Import latest player data as a json
get_json('json/fpl_events.json', 'https://fantasy.premierleague.com/api/bootstrap-static/')
# Open the json file
with open('json/fpl_events.json') as json_data:
    d = json.load(json_data)

# Get last GW if we don't specify a value for it
last_GW = get_latest_GW(d)
next_GW = last_GW + 1

# Machine learning hyperparameters
n_layers=2
n_neurons = 110
learn_rate = 0.00003
drop_rate = 0.2
patience_ = 20
batch_norm = True
factor_ = 0.7