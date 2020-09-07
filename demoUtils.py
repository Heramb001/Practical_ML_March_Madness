import pandas as pd
import numpy as np

# Function that returns the seeds for every passed seed with region
def get_seed(x):
    return int(x[1:3])

# Function that returns the dataset with onl seeddifference values in it ofr feature engineering.
def getSeedDataSet(tcr_df, tseeds_df):
    tourResults = tcr_df.copy()
    tourResults = tourResults[['Season','DayNum','WTeamID','WScore','LTeamID','LScore','WLoc','NumOT']]
    tourResults.rename(columns={'WTeamID':'T1', 'LTeamID':'T2', 'WScore':'ScoreT1', 'LScore':'ScoreT2'}, inplace=True)
    print('Total samples : ',len(tourResults))
    
    #--- get both the seeds into place
    tourney_result = pd.merge(tourResults, tseeds_df, left_on=['Season', 'T1'], right_on=['Season', 'TeamID'], how='left')
    tourney_result.rename(columns={'Seed':'T1Seed'}, inplace=True)
    tourney_result = tourney_result.drop('TeamID', axis=1)
    tourney_result = pd.merge(tourney_result, tseeds_df, left_on=['Season', 'T2'], right_on=['Season', 'TeamID'], how='left')
    tourney_result.rename(columns={'Seed':'T2Seed'}, inplace=True)
    tourney_result = tourney_result.drop('TeamID', axis=1)
    tourney_result['T1Seed'] = tourney_result['T1Seed'].map(lambda x: get_seed(x))
    tourney_result['T2Seed'] = tourney_result['T2Seed'].map(lambda x: get_seed(x))

    #--- winning teams
    tourney_win_result = tourney_result.drop(['Season', 'T1', 'T2'], axis=1)
    tourney_win_result['SeedDiff'] = tourney_win_result['T1Seed'] - tourney_win_result['T2Seed']
    tourney_win_result['result'] = 1

    #--- losing Teams
    tourney_lose_result = tourney_result.drop(['Season', 'T1', 'T2'], axis=1)
    tourney_lose_result['T1Seed'] = tourney_win_result['T2Seed']
    tourney_lose_result['T2Seed'] = tourney_win_result['T1Seed']
    tourney_lose_result['ScoreT1'] = tourney_win_result['ScoreT2']
    tourney_lose_result['ScoreT2'] = tourney_win_result['ScoreT1']
    tourney_lose_result['SeedDiff'] = tourney_lose_result['T1Seed'] - tourney_lose_result['T2Seed']
    tourney_lose_result['result'] = 0
    #--- creating a training dataset
    train_df = pd.concat((tourney_win_result, tourney_lose_result)).reset_index(drop=True)
    print('Total Dataset created of samples : ',len(train_df))
    #--- return the winning and losing dataset
    return train_df
