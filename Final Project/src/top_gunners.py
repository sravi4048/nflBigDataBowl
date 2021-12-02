import matplotlib.pyplot as plt
from pandas.io.pickle import read_pickle
import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import LinearRegression, LogisticRegression
import statsmodels.api as sm
import math
import xgboost as xgb
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, ShuffleSplit
from sklearn.metrics import f1_score, classification_report, confusion_matrix

def identifyGunners():
    #print(sample_tracking[(sample_tracking['gameId'] == 2018090600) & (sample_tracking['playId'] == 366) & (sample_tracking['jerseyNumber'] == 18.0)])
    scouting = pd.read_pickle("Data/scoutingdata.pkl")
    plays = pd.read_pickle("Data/plays.pkl")
    games = pd.read_pickle("Data/games.pkl")
    punt_plays_scouting = scouting.merge(plays, on=['gameId', 'playId'])
    punt_plays_scouting = punt_plays_scouting.merge(games, how='outer', on=['gameId'])
    punt_plays_scouting = punt_plays_scouting[punt_plays_scouting['specialTeamsPlayType'] == 'Punt']
    cols = ['gameId', 'playId', 'homeTeamAbbr', 'visitorTeamAbbr', 'operationTime', 'hangTime', 'kickType', 'kickDirectionIntended', 'kickDirectionActual', 
    'missedTackler', 'assistTackler', 'tackler', 'gunners', 'specialTeamsResult', 'playResult', 'kickReturnYardage']
    punt_plays_scouting = punt_plays_scouting[cols]
    print(len(punt_plays_scouting))
    test = punt_plays_scouting.head(20)
    gunner_ids = []
    '''
    print(len(comb))
    '''
    for i in range(len(punt_plays_scouting)):
        val = punt_plays_scouting.iloc[i]
        gunner_val = str(punt_plays_scouting.iloc[i]['gunners'])
        gameId = val['gameId']
        playId = val['playId']
        gunners = gunner_val.split(';')
        gunners = [gunner.strip() for gunner in gunners]
        for gunner in gunners:
            if gunner != 'nan':
                if gunner[:3] == val['homeTeamAbbr']:
                    team = 'home'
                else:
                    team = 'away'
                jersey_num = float(gunner[3:].strip())
                gunner_ids.append((gameId, playId, team, jersey_num))
            #gunner_ids.append((gameId, playId, comb[(comb['team'] == team) & (comb['gameId'] == gameId) & \
             #   (comb['playId'] == playId) & (comb['jerseyNumber'] == jersey_num)]))
        #print(gunners)
    #print(gunner_ids)
    gunners = pd.DataFrame(gunner_ids, columns=['gameId', 'playId', 'team', 'jerseyNumber'])
    print(gunners.head())
    print(gunners.columns)
    tracking_df_2018 = pd.read_pickle("Data/2018trackingdata.pkl")
    tracking_df_2018[(tracking_df_2018['gameId'].isin(punt_plays_scouting['gameId'])) & (tracking_df_2018['playId'].isin(punt_plays_scouting['playId']))]
    tracking_df_2019 = pd.read_pickle("Data/2019trackingdata.pkl")
    tracking_df_2019[(tracking_df_2019['gameId'].isin(punt_plays_scouting['gameId'])) & (tracking_df_2019['playId'].isin(punt_plays_scouting['playId']))]
    tracking_df_2020 = pd.read_pickle("Data/2020trackingdata.pkl")
    tracking_df_2020[(tracking_df_2020['gameId'].isin(punt_plays_scouting['gameId'])) & (tracking_df_2020['playId'].isin(punt_plays_scouting['playId']))]
    comb = pd.concat([tracking_df_2018, tracking_df_2019, tracking_df_2020])
    del tracking_df_2018
    del tracking_df_2019
    del tracking_df_2020
    print(len(comb))
    gunner_df = gunners.merge(comb, how='inner', on=['gameId', 'playId', 'team', 'jerseyNumber'])
    print(len(gunner_df))
    print(gunner_df.columns)
    gunner_df.to_pickle('Data/gunner_tracking.pkl')

def puntRegressionAnalysis():
    scouting = pd.read_pickle("Data/scoutingdata.pkl")
    plays = pd.read_pickle("Data/plays.pkl")
    punt_plays_scouting = scouting.merge(plays, on=['gameId', 'playId'])
    punt_plays_scouting = punt_plays_scouting[punt_plays_scouting['specialTeamsPlayType'] == 'Punt']
    #print(punt_plays_scouting.columns)
    cols = ['gameId', 'playId', 'operationTime', 'hangTime', 'kickLength', 'kickType', 'kickDirectionIntended', 'kickDirectionActual', 
    'missedTackler', 'assistTackler', 'tackler', 'gunners', 'specialTeamsResult', 'playResult', 'kickReturnYardage', 'preSnapHomeScore', 'preSnapVisitorScore', 
    'absoluteYardlineNumber', 'kickContactType', 'vises']
    punt_plays_scouting = punt_plays_scouting[cols]
    punt_plays_scouting['kickDirectionMatches'] = (punt_plays_scouting['kickDirectionIntended'] == punt_plays_scouting['kickDirectionActual']).astype(int)
    punt_plays_scouting['kickTypeNormal'] = (punt_plays_scouting['kickType'] == 'N').astype(int)
    punt_plays_scouting['numGunners'] = punt_plays_scouting['gunners'].apply(lambda x: str(x).count(';') + 1 if x == x else 0)
    #plt.hist(punt_plays_scouting['numGunners'].values)
    #plt.show()
    #return
    punt_plays_scouting['numMissedTackles'] = punt_plays_scouting['missedTackler'].apply(lambda x: str(x).count(';') + 1 if x == x else 0)
    punt_plays_scouting['numVises'] = punt_plays_scouting['vises'].apply(lambda x: str(x).count(';') + 1 if x == x else 0)
    punt_plays_scouting['totalTime'] = punt_plays_scouting['hangTime'] + punt_plays_scouting['operationTime']
    #print(punt_plays_scouting[['gunners', 'numGunners']].head(20))
    punt_plays_scouting = pd.get_dummies(punt_plays_scouting, prefix='contacttype', columns=['kickContactType'])
    punt_plays_scouting['gunnersVsVises'] = punt_plays_scouting['numGunners'] - punt_plays_scouting['numVises']
    print(punt_plays_scouting.head(20))
    print(punt_plays_scouting.columns.tolist())
    pred_cols = ['totalTime', 'kickLength', 'kickTypeNormal', 'numVises', 'numGunners',
    'contacttype_BB', 'contacttype_BC', 'contacttype_BF', 'contacttype_BOG', 'contacttype_CC', 'contacttype_CFFG', 
    'contacttype_DEZ', 'contacttype_ICC', 'contacttype_KTB', 'contacttype_KTC', 'contacttype_KTF', 'contacttype_MBC', 
    'contacttype_MBDR', 'contacttype_OOB']
    punt_plays_scouting = punt_plays_scouting.dropna(subset=pred_cols)
    X = punt_plays_scouting[pred_cols].values
    Y = punt_plays_scouting['playResult'].values
    reg = LinearRegression().fit(X, Y)
    print(reg.coef_)
    print(reg.intercept_)
    print(reg.score(X, Y))
    plt.clf()
    plt.scatter(Y, reg.predict(X) - Y)
    plt.title("Residuals plot")
    plt.xlabel("X")
    plt.ylabel("Residual")
    #plt.show()
    plt.clf()
    plt.scatter(X[:, 4], Y)
    plt.title("Scatter Missed Tackles vs Result")
    #plt.show()
    plt.clf()
    #plt.scatter(X[:, 0], reg.predict(X))
    #plt.show()

    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit() 
    print("")
    print(model.params)
    print(model.tvalues)
    print(model.bse)
    print(model.pvalues)

def identifyReturners():
    plays = pd.read_pickle("Data/combplaydata.pkl")
    returner_df = plays[plays['specialTeamsPlayType'] == 'Punt'][['gameId', 'playId', 'specialTeamsResult', 'returnerId']]
    returner_df = returner_df[~(returner_df.returnerId.str.contains(';', na=True))]
    returner_df = returner_df.rename(columns={"returnerId": "nflId"})
    returner_df['nflId'] = pd.to_numeric(returner_df['nflId'])
    returner_df = returner_df.dropna(subset=['nflId'])
    print((returner_df['specialTeamsResult'] == 'Fair Catch').astype(int).sum())
    print(returner_df.dtypes)
    return
    tracking_df_2018 = pd.read_pickle("Data/2018trackingdata.pkl")
    #tracking_df_2018 = tracking_df_2018[(tracking_df_2018['gameId'].isin(returner_df['gameId'])) & (tracking_df_2018['playId'].isin(returner_df['playId']))]
    print('2018 loaded')
    tracking_df_2019 = pd.read_pickle("Data/2019trackingdata.pkl")
    #tracking_df_2019 = tracking_df_2019[(tracking_df_2019['gameId'].isin(returner_df['gameId'])) & (tracking_df_2019['playId'].isin(returner_df['playId']))]
    print('2019 loaded')
    tracking_df_2020 = pd.read_pickle("Data/2020trackingdata.pkl")
    #tracking_df_2020[(tracking_df_2020['gameId'].isin(returner_df['gameId'])) & (tracking_df_2020['playId'].isin(returner_df['playId']))]
    print('2020 loaded')
    comb = pd.concat([tracking_df_2018, tracking_df_2019, tracking_df_2020])
    print('concat done')
    del tracking_df_2018
    del tracking_df_2019
    del tracking_df_2020
    print('deleted')
    returner_df = returner_df.merge(comb, how='inner', on=['gameId', 'playId', 'nflId'])
    print(len(returner_df))
    print(returner_df.columns.tolist())
    #returner_df.to_pickle('Data/punt_returner_tracking.pkl')

def test(result):
    '''
    returners_df = pd.read_pickle('Data/punt_returner_tracking.pkl')
    gunners_df = pd.read_pickle('Data/gunner_tracking.pkl')
    gameId = 2018090905
    playId = 2393
    print(returners_df.query('@gameId == gameId and @playId == playId').shape[0])
    print(gunners_df.query('@gameId == gameId and @playId == playId').shape[0])
    print((gameId in set(returners_df['gameId'])) and (playId in set(returners_df['playId'])))
    print((gameId in set(gunners_df['gameId'])) and (playId in set(gunners_df['playId'])))
    '''
    plays = pd.read_pickle("Data/combplaydata.pkl")
    returner_df = pd.read_pickle("Data/punt_returner_tracking.pkl")
    #playkey = '2019122900_393'
    pd.set_option('display.max_rows', 100)
    print(plays.loc[plays['playkey'].isin(result)][['playkey', 'specialTeamsPlayType', 'specialTeamsResult', 'returnerId', 'gunners', 'kickLength', 'kickReturnYardage', 'playResult']].head(100))
    try:
        print(returner_df.loc[returner_df['playkey'] == playkey].iloc[0])
    except:
        pass
    pd.set_option('display.max_rows', 10)

def gunnerRegression():
    '''
    gunner_speed_df['beforePuntCaught'] = gunner_speed_df.apply(lambda x: 1 if (x.frameId <= punt_received_df.loc[(punt_received_df.playkey == x.playkey) & (punt_received_df.nflId_gunner == x.nflId_gunner)].frameId).item() else 0, axis=1)
    print(len(gunner_speed_df))
    gunner_speed_df = gunner_speed_df[gunner_speed_df['beforePuntCaught'] == 1]
    gunner_speed_df = gunner_speed_df.drop(['beforePuntCaught'], axis=1)
    print(len(gunner_speed_df))
    return
    gunner_speed_df = punt_received_df.groupby(by=['gameId', 'playId', 'nflId_gunner'])[['s_gunner']].max().reset_index()
    gunner_speed_df = gunner_speed_df.rename(columns={'s_gunner': 'max_speed'})
    #print(gunner_speed_df.head(20))
    '''

    df = pd.read_pickle("Data/processed_gunner_returner_df.pkl")
    df1 = df.groupby(by=['gameId', 'playId'])[['distance']].min().reset_index()
    df = df1.merge(df, on=['gameId', 'playId', 'distance'])
    
    #df['distanceSquared'] = (df['distance']) ** 2
    #print(len(df))
    X = df[['distance', 'hangTime', 'kickLength', 'a_returner', 'returner_movement_forward', 'returner_distance_from_endzone_less_than10']].values
    Y = df['fairCatch'].values
    reg = LogisticRegression(random_state=0).fit(X, Y)
    print(reg.coef_)
    print(reg.intercept_)
    print(reg.score(X, Y))
    plt.scatter(X[:, 3], Y)
    #plt.show()

    X = sm.add_constant(X)
    model = sm.Logit(Y, X).fit() 
    print(model.summary())
    print(model.pvalues)
    
    X = df[['distance', 'hangTime', 'kickLength', 'returner_movement_forward', 'a_returner']].values
    Y = df['kickReturnYardage'].values
    reg = LinearRegression().fit(X, Y)
    print(reg.coef_)
    print(reg.intercept_)
    print(reg.score(X, Y))
    plt.clf()
    plt.scatter(X[:, 1], reg.predict(X) - Y)
    plt.title("Residuals plot")
    plt.xlabel("X")
    plt.ylabel("Residual")
    #plt.show()
    plt.clf()
    plt.scatter(X[:, 0], Y)
    plt.title("Scatter Distance vs Result")
    #plt.show()
    plt.clf()
    
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit() 
    print("")
    #print(model.params)
    #print(model.tvalues)
    #print(model.bse)
    print(model.summary())

    X = df[['hangTime', 'kickLength', 'returner_movement_forward']].values
    Y = df['distance'].values

    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit() 
    print("")
    #print(model.params)
    #print(model.tvalues)
    #print(model.bse)
    print(model.summary())



    '''
    plt.hist(df['kickReturnYardage'], bins=50)
    plt.xlabel('Punt Return Yards')
    plt.ylabel('Frequency')
    plt.show()
    '''
def startingPositionDistances():
    punt_plays_scouting = pd.read_pickle("Data/combplaydata.pkl")
    punt_plays_scouting = punt_plays_scouting[punt_plays_scouting['specialTeamsPlayType'] == 'Punt']
    test = punt_plays_scouting.head(20)
    #for y in test.loc[test['specialTeamsPlayType'] == 'Punt'].itertuples():
    #   print(y)
    #   print(type(y))
    tracking_df_2018 = pd.read_pickle("Data/2018trackingdata.pkl")
    tracking_df_2018 = tracking_df_2018[(tracking_df_2018['gameId'].isin(punt_plays_scouting['gameId'])) & (tracking_df_2018['playId'].isin(punt_plays_scouting['playId']))]
    tracking_df_2019 = pd.read_pickle("Data/2019trackingdata.pkl")
    tracking_df_2019 = tracking_df_2019[(tracking_df_2019['gameId'].isin(punt_plays_scouting['gameId'])) & (tracking_df_2019['playId'].isin(punt_plays_scouting['playId']))]
    tracking_df_2020 = pd.read_pickle("Data/2020trackingdata.pkl")
    tracking_df_2020 = tracking_df_2020[(tracking_df_2020['gameId'].isin(punt_plays_scouting['gameId'])) & (tracking_df_2020['playId'].isin(punt_plays_scouting['playId']))]
    comb = pd.concat([tracking_df_2018, tracking_df_2019, tracking_df_2020])
    print(len(comb))
    del tracking_df_2018
    del tracking_df_2019
    del tracking_df_2020
    gunner_df = pd.read_pickle('Data/gunner_tracking.pkl')
    comb = comb[comb['frameId'] == 1]
    starting_position_df = gunner_df[gunner_df['event'] == 'ball_snap']
    #print(comb.head(50))
    starting_position_df['Distances'] = starting_position_df.apply(lambda x: [math.sqrt((x.x - y.x) ** 2 + (x.y - y.y) ** 2) for y in comb.loc[(comb['gameId'] == x.gameId) & (comb['playId'] == x.playId)].itertuples()], axis=1)
    #print(test[['gameId', 'playId', 'nflId', 'Distances', 'frameId']].head(20))
    starting_position_df.to_pickle('Data/gunners_distances.pkl')

def preprocessGunnersReturnersdf():
    ## Get punt plays from df
    plays = pd.read_pickle("Data/combplaydata.pkl")
    punt_df = plays[plays['specialTeamsPlayType'] == 'Punt']
    returners_df = pd.read_pickle('Data/punt_returner_tracking.pkl')
    gunners_df = pd.read_pickle('Data/gunner_tracking.pkl')

    ## Merge returner and gunner dfs
    merged_df = gunners_df.merge(returners_df, how='inner', on=['playkey', 'gameId', 'playId', 'time', 'frameId', 'event', 'playDirection'], suffixes=["_gunner", "_returner"], validate="many_to_one")
    punt_received_df = merged_df[merged_df.event.isin(['punt_received', 'fair_catch', 'punt_land'])]
    punt_received_df = punt_received_df.drop_duplicates(subset=['playkey', 'nflId_gunner'])

    punt_df = punt_df[['playkey', 'gameId', 'playId', 'specialTeamsResult', 'kickLength', 'returnerId', 'kickReturnYardage', 'playResult', 'hangTime', 'operationTime']]
    punt_received_df = punt_received_df.merge(punt_df, how='inner', on=['gameId', 'playId', 'playkey'])
    #print(len(punt_received_df))

    ##Get stats from 5 frames before punt received for better fair catch prediction
    copy_df = punt_received_df.copy()
    copy_df['frameId'] = punt_received_df['frameId'].apply(lambda x: x - 5)
    copy_df = copy_df[['playkey', 'gameId', 'playId', 'frameId', 'nflId_gunner']]
    #print(punt_received_df['frameId'].head(5), copy_df['frameId']. head(5))
    print(copy_df.columns.tolist())
    #print(merged_df.columns.tolist())
    five_frames_back_df = copy_df.merge(merged_df, on=['playkey', 'gameId', 'playId', 'frameId', 'nflId_gunner'])
    five_frames_back_df = five_frames_back_df[['playkey', 'gameId', 'playId', 'frameId', 'nflId_gunner', 'x_gunner', 'y_gunner', 'x_returner', 'y_returner', 's_gunner', 'a_gunner', 'dir_returner', 's_returner', 'a_returner']]
    #print(five_frames_back_df.head(5))
    #print(len(punt_received_df))
    punt_received_df = punt_received_df.merge(five_frames_back_df, on=['playkey', 'gameId', 'playId', 'nflId_gunner'], suffixes=[None, '_5back'])  
    #print(len(punt_received_df))
    #print(five_frames_back_df.head(20))
    print(punt_received_df.columns.tolist())
    print(punt_received_df.head(10))

    ## Create new variables for regression
    punt_received_df['fairCatch'] = (punt_received_df['specialTeamsResult'] == 'Fair Catch').astype(int)
    #print(punt_received_df['fairCatch'].sum())
    punt_received_df['distance_at_catch'] = punt_received_df.apply(lambda x: math.sqrt(((x.x_gunner - x.x_returner) ** 2) + ((x.y_gunner - x.y_returner)  ** 2)), axis=1)
    punt_received_df['distance_5back'] = punt_received_df.apply(lambda x: math.sqrt(((x.x_gunner_5back - x.x_returner_5back) ** 2) + ((x.y_gunner_5back - x.y_returner_5back)  ** 2)), axis=1)
    punt_received_df['returner_distance_from_endzone'] = punt_received_df.apply(lambda x: 50 - abs(60 - x.x_returner), axis=1)
    punt_received_df['returner_distance_from_endzone_5back'] = punt_received_df.apply(lambda x: 50 - abs(60 - x.x_returner_5back), axis=1)
    punt_received_df['returner_distance_from_endzone_less_than10'] = punt_received_df.apply(lambda x: 1 if x.returner_distance_from_endzone < 10 else 0, axis=1)
    punt_received_df['returner_distance_from_endzone_less_than10_5back'] = punt_received_df.apply(lambda x: 1 if x.returner_distance_from_endzone_5back < 10 else 0, axis=1)
    #punt_received_df['absoluteYardline'] = punt_received_df
    punt_received_df['returner_movement_forward'] = punt_received_df.apply(lambda x: 1 if ((180 < x.dir_returner and x.playDirection == 'right') or (180 > x.dir_returner and x.playDirection == 'left')) else 0, axis=1)
    punt_received_df['returner_movement_forward_5back'] = punt_received_df.apply(lambda x: 1 if ((180 < x.dir_returner_5back and x.playDirection == 'right') or (180 > x.dir_returner_5back and x.playDirection == 'left')) else 0, axis=1)
    punt_received_df['kickReturnYardage'] = punt_received_df['kickReturnYardage'].fillna(0)
    punt_received_df['return_not_positive'] = (punt_received_df['kickReturnYardage'] <= 0).astype(int)
    #(punt_received_df['x_gunner'] ** 2) + (punt_received_df['y_gunner'] ** 2)
    #print(punt_received_df.head(20))
    #print(len(punt_received_df))

    ## Merge dfs for regression
    #df = df.merge(gunner_speed_df, on=['gameId', 'playId', 'nflId_gunner'])
    #print(df.head(20))

    punt_received_df = punt_received_df.dropna(subset=['distance_at_catch', 'kickLength', 'hangTime', 'operationTime'])
    punt_received_df.to_pickle("Data/processed_gunner_returner_df.pkl")
    return punt_received_df

def oldTopGunners(when="5"):
    if when == "catch":
        when = "distance_at_catch"
        app = ""
    elif when == "5":
        when = "distance_5back"
        app = "_5back"
    #Load starting positions to see double_teamed
    starting_position_df = pd.read_pickle('Data/gunners_distances.pkl')
    starting_position_df['closest_other'] = starting_position_df.apply(lambda x: sorted(x.Distances)[1], axis=1)
    starting_position_df['second_closest'] = starting_position_df.apply(lambda x: sorted(x.Distances)[2], axis=1)
    
    starting_position_df['double_teamed'] = (starting_position_df['second_closest'] < 5).astype(int)
    view = starting_position_df[['gameId', 'playId', 'nflId', 'Distances', 'frameId', 'closest_other', 'second_closest', 'double_teamed']]
    punt_received_df = pd.read_pickle("Data/processed_gunner_returner_df.pkl")
    comb = punt_received_df.merge(starting_position_df, left_on=['gameId', 'playId', 'nflId_gunner', 'displayName_gunner'], right_on=['gameId', 'playId', 'nflId', 'displayName'])
    '''
    df1 = punt_received_df.groupby(by=['gameId', 'playId'])[[when]].min().reset_index()
    punt_received_df = df1.merge(punt_received_df, on=['gameId', 'playId', when])
    #print(comb.head(10))

    X_log = comb[[when, 'hangTime', 'kickLength', 'a_returner'+app, 'returner_movement_forward'+app, 'returner_distance_from_endzone_less_than10'+app]].values
    Y_log = comb['fairCatch'].values
    train_x_log, test_x_log, train_y_log, test_y_log = train_test_split(X_log, Y_log, test_size=0.2)
    reg = xgb.XGBClassifier(n_estimators=20, tree_method="hist", eval_metric='logloss')
    reg.fit(train_x_log, train_y_log, eval_set=[(test_x_log, test_y_log)])
    preds = reg.predict(test_x_log)
    print(metrics.f1_score(test_y_log, preds))
    xgb.plot_importance(reg)
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.show()

    #X = df[['distance', 'hangTime', 'kickLength', 'returner_movement_forward', 'a_returner']].values
    X_lin = comb[[when, 'hangTime', 'kickLength', 'a_returner'+app, 'returner_movement_forward'+app,]].values
    Y_lin = comb['kickReturnYardage'].values
    train_x_lin, test_x_lin, train_y_lin, test_y_lin = train_test_split(X_lin, Y_lin, test_size=0.2)
    reg = xgb.XGBRegressor(n_estimators=20, tree_method="hist", eval_metric=metrics.mean_absolute_error)
    reg.fit(train_x_lin, train_y_lin, eval_set=[(test_x_lin, test_y_lin)])
    print(reg.score(test_x_lin, test_y_lin))

    log_reg = LogisticRegression(random_state=0).fit(train_x_log, train_y_log)
    preds = log_reg.predict(test_x_log)
    print(metrics.f1_score(test_y_log, preds))
    print(metrics.confusion_matrix(log_reg.predict(X_log), Y_log))
    '''
    comb['totalTime'] = comb.apply(lambda x: x.hangTime + x.operationTime, axis=1)
    comb = comb[['operationTime', 'displayName_gunner', 'hangTime', 'kickLength', 'returner_movement_forward'+app, 'double_teamed', when, 'kickReturnYardage', 'totalTime', 
    's_returner'+app, 'a_returner'+app, 'returner_distance_from_endzone_less_than10'+app, 'return_not_positive', 'fairCatch']]
    X_log = comb[[when, 'totalTime', 'kickLength', 'a_returner'+app, 'returner_movement_forward'+app, 'returner_distance_from_endzone_less_than10'+app]].values
    Y_log = comb['fairCatch'].values
    #train_x_log, test_x_log, train_y_log, test_y_log = train_test_split(X_log, Y_log, test_size=0.2)
    ## Gunner Performance Regression
    X_lin = comb[['kickLength', 'double_teamed', 'returner_movement_forward'+app, 'totalTime']].values
    Y_lin = comb[when].values

    reg = LinearRegression().fit(X_lin, Y_lin)
    #print(reg.coef_)
    #print(reg.intercept_)
    #print(reg.score(X_lin, Y_lin))

    log_reg = LogisticRegression(random_state=0).fit(X_log, Y_log)
    print(log_reg.score(X_log, Y_log))
    print(metrics.confusion_matrix(log_reg.predict(X_log), Y_log))

    test = sm.add_constant(X_log)
    model = sm.OLS(Y_log, test).fit() 
    #print("")
    print(model.summary())
    ## Top gunners
    preds = reg.predict(X_lin)
    comb['preds'] = pd.Series(preds)
    if when == "distance_at_catch":
        comb['distanceCloser'] = comb.apply(lambda x: x.preds - x.distance_at_catch, axis=1)
    elif when == "distance_5back":
        comb['distanceCloser'] = comb.apply(lambda x: x.preds - x.distance_5back, axis=1)
    fc_prob = log_reg.predict_proba(X_log)
    comb['fairCatchPredictedProb'] = pd.Series(fc_prob[:, 0])
    X_log = comb[['preds', 'totalTime', 'kickLength', 'a_returner'+app, 'returner_movement_forward'+app, 'returner_distance_from_endzone_less_than10'+app]].values
    pred_prob_based_on_pred_distance = log_reg.predict_proba(X_log)
    prob_effect = fc_prob - pred_prob_based_on_pred_distance
    #return
    comb['probEffect'] = pd.Series(prob_effect[:, 1])
    df = comb.groupby(by=['displayName_gunner'])[['hangTime', 'double_teamed', 'distanceCloser', when, 'preds', 'kickReturnYardage', 'fairCatchPredictedProb', 'probEffect']].agg(['mean', 'count']).reset_index()

    df = df.sort_values(by=[('probEffect', 'mean')], ascending=False)
    df = df[df[(when, 'count')] > 30]
    df = df[[(when, 'count'), ('distanceCloser', 'mean'), ('displayName_gunner',''), (when, 'mean'), ('double_teamed', 'mean'), 
    ('hangTime', 'mean'), ('kickReturnYardage', 'mean'), ('fairCatchPredictedProb', 'mean'), ('probEffect', 'mean')]]
    print(len(df))
    print(df.head(20))
    df.to_csv('topGunners.csv')

def xgbModelPrep():
    
    plays = pd.read_pickle("Data/combplaydata.pkl")
    punt_df = plays[plays['specialTeamsPlayType'] == 'Punt']
    returners_df = pd.read_pickle('Data/punt_returner_tracking.pkl')
    gunners_df = pd.read_pickle('Data/gunner_tracking.pkl')
    punt_df['numGunners'] = punt_df['gunners'].apply(lambda x: str(x).count(';') + 1 if x == x else 0)
    punt_df = punt_df[punt_df['numGunners'] == 2]
    punt_df = punt_df[['playkey', 'gameId', 'playId', 'specialTeamsResult', 'kickLength', 'returnerId', 'kickReturnYardage', 'playResult', 'hangTime', 'operationTime']]
    for col in gunners_df.columns.tolist():
        if col[0] == 'x':
            gunners_df[col] = gunners_df.apply(lambda x: x[col] if x['playDirection'] == 'right' else 120 - x[col], axis=1)
        elif col[0] == 'o' or col[:3] == 'dir':
            gunners_df[col] = gunners_df.apply(lambda x: 360 - x[col], axis=1)
    gunners_df = gunners_df[gunners_df['playkey'].isin(punt_df['playkey'])]
    for col in returners_df.columns.tolist():
        if col[0] == 'x':
            returners_df[col] = returners_df.apply(lambda x: x[col] if x['playDirection'] == 'right' else 120 - x[col], axis=1)
        elif col[0] == 'o' or col[:3] == 'dir':
            returners_df[col] = returners_df.apply(lambda x: 360 - x[col], axis=1)
    returners_df = returners_df[returners_df['playkey'].isin(punt_df['playkey'])]
    first_gunners_df = gunners_df.drop_duplicates(subset=['playkey', 'gameId', 'playId', 'time', 'frameId', 'event', 'playDirection', 'team'], keep='first')
    second_gunners_df = gunners_df.drop_duplicates(subset=['playkey', 'gameId', 'playId', 'time', 'frameId', 'event', 'playDirection', 'team'], keep='last')
    #print(len(returners_df))
    #print(len(gunners_df))
    print(len(first_gunners_df))
    print(len(second_gunners_df))
    #print(first_gunners_df.head(10))
    #print(second_gunners_df.head(20))
    gunners_df = first_gunners_df.merge(second_gunners_df, on=['playkey', 'gameId', 'playId', 'time', 'frameId', 'event', 'playDirection', 'team'], suffixes=[None, "_two"])
    #gunners_df = gunners_df.join(gunners_df, rsuffix="_two")
    #on=['playkey', 'gameId', 'playId', 'time', 'frameId', 'event', 'playDirection', 'team']
    pd.set_option('display.max_rows', 150)
    #print(gunners_df.head(150))
    print(gunners_df.iloc[0])
    print(gunners_df.iloc[1])
    pd.set_option('display.max_rows', 10)

    print(gunners_df.columns.tolist())
    print(gunners_df.head(10))
    ## Merge returner and gunner dfs
    merged_df = gunners_df.merge(returners_df, how='inner', on=['playkey', 'gameId', 'playId', 'time', 'frameId', 'event', 'playDirection'], suffixes=["_one", "_returner"])
    print(merged_df.columns.tolist())
    punt_received_df = merged_df[merged_df.event.isin(['punt_received', 'fair_catch'])]
    print(len(merged_df))
    merged_df = merged_df.drop_duplicates(subset=['playkey', 'nflId_one', 'nflId_two', 'nflId_returner', 'frameId'])
    print(len(merged_df))
    
    merged_df.to_pickle('Data/two_gunner_plays_tracking.pkl')
    return
def xgbmodelPrep2():
    merged_df = pd.read_pickle('Data/two_gunner_plays_tracking.pkl')
    #print(merged_df.columns.tolist())
    #print(merged_df.head(20))
    punt_received_df = merged_df[merged_df.event.isin(['punt_received', 'fair_catch'])]
    #punt_received_df['duplicates'] = punt_received_df.duplicated(subset=['playkey'], keep=False)
    #print(punt_received_df.columns.tolist())
    #print(punt_received_df[punt_received_df['duplicates']][['playkey', 'team_one', 'time', 'event', 'nflId_one', 'frameId', 'playDirection', 'jerseyNumber_two', 'nflId_two', 'displayName_two', 'position_two', 'nflId_returner', 'displayName_returner', 'jerseyNumber_returner']])
    punt_received_df = punt_received_df.drop_duplicates(subset=['playkey'])
    punt_received_df = punt_received_df.set_index(keys='playkey', verify_integrity=True)
    print(len(merged_df))
    merged_df = merged_df[merged_df.playkey.isin(punt_received_df.index)]
    print(merged_df.event.unique())
    ball_snapped_df = merged_df[merged_df.event.isin(['ball_snap'])]
    merged_df['secondsSinceSnap'] = merged_df.apply(lambda x: x['frameId'] / 10 if x['playkey'] not in ball_snapped_df['playkey'] else (x['frameId'] - ball_snapped_df.loc[x['playkey'], 'frameId'].item())/10, axis=1)
    merged_df.to_pickle('Data/two_gunner_plays_tracking.pkl')
    merged_df['before punt received'] = merged_df.apply(lambda x: x['frameId'] < punt_received_df.loc[x['playkey'], 'frameId'].item(), axis=1)
    #pd.set_option('display.max_rows', 151)
    #print(merged_df.head(150))
    print(len(merged_df))
    merged_df = merged_df[merged_df['before punt received']]
    print(len(merged_df))
    merged_df.to_pickle('Data/two_gunner_plays_before_catch.pkl')
    #pd.set_option('display.max_rows', 10)
    return
    
    punt_received_df = punt_received_df.merge(punt_df, how='inner', on=['gameId', 'playId', 'playkey'])
    #print(len(punt_received_df))

    ##Get stats from 5 frames before punt received for better fair catch prediction
    copy_df = punt_received_df.copy()
    copy_df['frameId'] = punt_received_df['frameId'].apply(lambda x: x - 5)
    copy_df = copy_df[['playkey', 'gameId', 'playId', 'frameId', 'nflId_gunner']]

def hyperparameterTuning(reg):
    #train_x_log, test_x_log, train_y_log, test_y_log = train_test_split(X_log, Y_log, test_size=0.2) # edit this so that different plays aren't split up
    #params = { 'max_depth': [6],
    #       'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05],
    #       'n_estimators': [1000],
    #       'colsample_bytree': [0.7],
    #       'tree_method':['hist']}
    #clf = GridSearchCV(estimator=reg, param_grid=params, scoring='f1', verbose=1)
    #clf.fit(train_x_log, train_y_log)
    #print("Best parameters:", clf.best_params_)
    #print("Highest F1 Score: ", (clf.best_score_))
    pass

def main():
    '''
    starting_position_df = pd.read_pickle('Data/gunners_distances.pkl')
    starting_position_df['closest_other'] = starting_position_df.apply(lambda x: sorted(x.Distances)[1], axis=1)
    starting_position_df['second_closest'] = starting_position_df.apply(lambda x: sorted(x.Distances)[2], axis=1)
    
    starting_position_df['double_teamed_one'] = (starting_position_df['second_closest'] < 5).astype(int)
    starting_position_df['double_teamed_two'] = (starting_position_df['second_closest'] < 5).astype(int)
    starting_position_df['nflId_one'] = starting_position_df['nflId']
    starting_position_df['nflId_two'] = starting_position_df['nflId']
    view_one = starting_position_df[['playkey', 'nflId_one', 'double_teamed_one']]
    view_two = starting_position_df[['playkey', 'nflId_two', 'double_teamed_two']]
    merged_df = pd.read_pickle('Data/two_gunner_plays_before_catch.pkl')
    print(len(merged_df))
    merged_df = merged_df.merge(view_one, on=['playkey', 'nflId_one'])
    print(len(merged_df))
    merged_df = merged_df.merge(view_two, on=['playkey', 'nflId_two'])
    print(len(merged_df))
    play_df = pd.read_pickle('Data/combplaydata.pkl')
    play_df['totalTime'] = play_df.apply(lambda x: x.hangTime + x.operationTime, axis=1)
    play_df['fairCatch'] = (play_df['specialTeamsResult'] == 'Fair Catch').astype(int)
    merged_df = merged_df.merge(play_df, on=['playkey'])
    print(len(merged_df))
    print(merged_df.head(10))
    print(merged_df.columns.tolist())
    merged_df['distance_one'] = merged_df.apply(lambda x: math.sqrt(((x.x_one - x.x_returner) ** 2) + ((x.y_one - x.y_returner)  ** 2)), axis=1)
    merged_df['distance_two'] = merged_df.apply(lambda x: math.sqrt(((x.x_two - x.x_returner) ** 2) + ((x.y_two - x.y_returner)  ** 2)), axis=1)
    merged_df.to_pickle('Data/two_gunner_plays_before_catch.pkl')
    return
    '''
    merged_df = pd.read_pickle('Data/two_gunner_plays_before_catch.pkl')
    merged_df['secondsSinceSnap'] = merged_df.apply(lambda x: (x['frameId'] - 11) / 10, axis=1)
    #print(merged_df.columns.tolist())
    play_df = pd.read_pickle('Data/combplaydata.pkl')
    play_df = play_df[play_df['playkey'].isin(merged_df.playkey)]
    #print(len(play_df))
    punt_received_df = merged_df.drop_duplicates(subset=['playkey'], keep='last')
    #print(len(punt_received_df))
    punt_received_df = punt_received_df.dropna(subset=['totalTime'])
    #print(len(punt_received_df))
    #print(punt_received_df[['playkey', 'frameId', 'nflId_one', 'nflId_two', 'nflId_returner']].head(20))
    #return
    train_plays, test_plays = train_test_split(play_df, test_size=0.4)
    pred_cols = ['distance_one', 'distance_two', 'totalTime', 'kickLength', 'x_one', 'y_one', 's_one', 'o_one', 'dir_one', 'x_two', 'y_two', 's_two', 'o_two', 'dir_two', 'x_returner', 'y_returner', 's_returner', 'a_returner', 'dir_returner', 'o_returner', 'secondsSinceSnap']
    train_x_log = punt_received_df[punt_received_df['playkey'].isin(train_plays['playkey'])][pred_cols]
    train_y_log = punt_received_df[punt_received_df['playkey'].isin(train_plays['playkey'])]['fairCatch']
    test_x_log = punt_received_df[punt_received_df['playkey'].isin(test_plays['playkey'])][pred_cols]
    test_y_log = punt_received_df[punt_received_df['playkey'].isin(test_plays['playkey'])]['fairCatch']
    X_log = merged_df[pred_cols]
    Y_log = merged_df['fairCatch']
    
    all_features_at_catch_reg = xgb.XGBClassifier(n_estimators=1000, max_depth=6, tree_method="hist", learning_rate=0.03, colsample_bytree=0.7, eval_metric='logloss')
    all_features_at_catch_reg.fit(train_x_log, train_y_log)
    preds = all_features_at_catch_reg.predict(test_x_log)
    #print("F1 Score of XGB Model on data at catch", metrics.f1_score(test_y_log, preds))
    #scores = cross_val_score(all_features_at_catch_reg, punt_received_df[pred_cols], punt_received_df['fairCatch'], cv=5, scoring='f1_macro')
    #print("Cross Val F1 Score of XGB Model with only catch data on data at catch: ", scores.mean())
    #scores = cross_val_score(all_features_at_catch_reg, merged_df[pred_cols], merged_df['fairCatch'], cv=5, scoring='f1_macro')
    #print("Cross Val F1 Score of XGB Model with only catch data on all data: ", scores.mean())
    #print(confusion_matrix(test_y_log, all_features_at_catch_reg.predict(test_x_log), labels=[1,0]))
    #print(classification_report(punt_received_df['fairCatch'], all_features_at_catch_reg.predict(X)))

    train_plays, test_plays = train_test_split(play_df, test_size=0.2)
    train_x_log = merged_df[merged_df['playkey'].isin(train_plays['playkey'])][pred_cols]
    train_y_log = merged_df[merged_df['playkey'].isin(train_plays['playkey'])]['fairCatch']
    #test_x_log = merged_df[merged_df['playkey'].isin(test_plays['playkey'])][pred_cols]
    #test_y_log = merged_df[merged_df['playkey'].isin(test_plays['playkey'])]['fairCatch']

    all_features_reg = xgb.XGBClassifier(n_estimators=1000, max_depth=6, tree_method="hist", learning_rate=0.03, colsample_bytree=0.7, eval_metric='logloss')
    all_features_reg.fit(train_x_log, train_y_log)
    preds = all_features_reg.predict(test_x_log)

    #print("F1 Score of XGB Model on all data tested on catch frames", metrics.f1_score(test_y_log, preds))
    #scores = cross_val_score(all_features_reg, punt_received_df[pred_cols], punt_received_df['fairCatch'], cv=5, scoring='f1_macro')
    print("F1 Score of Full XGB Model on data at catch: ", metrics.f1_score(test_y_log, all_features_reg.predict(test_x_log), average='macro'))
    #scores = cross_val_score(all_features_reg, merged_df[pred_cols], merged_df['fairCatch'], cv=5, scoring='f1_macro')
    #print("Cross Val F1 Score of Full XGB Model on all data: ", scores.mean())
    #print(confusion_matrix(test_y_log, all_features_reg.predict(test_x_log), labels=[1,0]))
    #print(classification_report(test_y_log, all_features_reg.predict(test_x_log)))
    #xgb.plot_importance(all_features_reg)
    #plt.rcParams['figure.figsize'] = [5, 5]
    #plt.show()
    xgb.plot_tree(all_features_reg,num_trees=999)
    plt.rcParams['figure.figsize'] = [50, 10]
    plt.show()

    merged_df = merged_df.dropna(subset=pred_cols)
    #print(merged_df[pred_cols].head())
    train_x_log = punt_received_df[punt_received_df['playkey'].isin(train_plays['playkey'])][pred_cols]
    train_y_log = punt_received_df[punt_received_df['playkey'].isin(train_plays['playkey'])]['fairCatch']
    log_reg = LogisticRegression(solver='liblinear', random_state=0, max_iter=1000).fit(train_x_log, train_y_log)
    scores = cross_val_score(log_reg, punt_received_df[pred_cols], punt_received_df['fairCatch'], cv=5, scoring='f1_macro')
    #print("Cross Val F1 Score of Logistic Model trained on data at catch: ", scores.mean())
    #scores = cross_val_score(log_reg, merged_df[pred_cols], merged_df['fairCatch'], cv=5, scoring='f1_macro')
    #print("Cross Val F1 Score of Logistic Model trained on data at catch on all data: ", scores.mean())

    #print(merged_df[pred_cols].head())
    merged_df = merged_df.dropna(subset=pred_cols)
    #print(merged_df[pred_cols].head())
    train_x_log = merged_df[merged_df['playkey'].isin(train_plays['playkey'])][pred_cols]
    train_y_log = merged_df[merged_df['playkey'].isin(train_plays['playkey'])]['fairCatch']
    #test_x_log = merged_df[merged_df['playkey'].isin(test_plays['playkey'])][pred_cols]
    #test_y_log = merged_df[merged_df['playkey'].isin(test_plays['playkey'])]['fairCatch']
    log_reg = LogisticRegression(solver='liblinear', random_state=0, max_iter=1000).fit(train_x_log, train_y_log)
    print("F1 Score of Full Logistic Model on data at catch: ", metrics.f1_score(test_y_log, log_reg.predict(test_x_log), average='macro'))

    log_reg = LogisticRegression(solver='liblinear', random_state=0, max_iter=1000).fit(train_x_log, train_y_log)
    scores = cross_val_score(log_reg, punt_received_df[pred_cols], punt_received_df['fairCatch'], cv=5, scoring='f1_macro')
    print("Cross Val F1 Score of Logistic Model trained on all data: ", scores.mean())
    scores = cross_val_score(log_reg, merged_df[pred_cols], merged_df['fairCatch'], cv=5, scoring='f1_macro')
    print("Cross Val F1 Score of Logistic Model trained on all data on all data: ", scores.mean())
    #preds = log_reg.predict(test_x_log)
    #print("F1 Score of Logistic Regression on all data tested on catch frames", metrics.f1_score(test_y_log, preds))
    #print(confusion_matrix(test_y_log, log_reg.predict(test_x_log), labels=[1,0]))
    #print(classification_report(test_y_log, log_reg.predict(test_x_log)))
    #xgb.plot_importance(all_features_at_catch_reg)
    #plt.rcParams['figure.figsize'] = [5, 5]
    #plt.clf()
    #xgb.plot_tree(all_features_at_catch_reg,num_trees=19)
    #plt.rcParams['figure.figsize'] = [50, 10]
    #plt.show()
    #plt.show()

    #null_data = train_x_log[train_x_log.isnull().any(axis=1)]
    #train_x_log = train_x_log.dropna(subset=['totalTime'])scaler = preprocessing.StandardScaler().fit(X_train)
    #scaler = preprocessing.StandardScaler().fit(train_x_log)
    #scaled_train_x_log = scaler.transform(train_x_log)
    '''
    log_reg = LogisticRegression(random_state=0).fit(train_x_log, train_y_log)
    preds = log_reg.predict(test_x_log)
    print(metrics.f1_score(test_y_log, preds))

    pred_cols = ['totalTime', 'kickLength', 'x_returner', 'y_returner', 's_returner', 'a_returner', 'dir_returner', 'o_returner', 'secondsSinceSnap']
    train_x_log = punt_received_df[punt_received_df['playkey'].isin(train_plays['playkey'])][pred_cols]
    train_y_log = punt_received_df[punt_received_df['playkey'].isin(train_plays['playkey'])]['fairCatch']
    test_x_log = punt_received_df[punt_received_df['playkey'].isin(test_plays['playkey'])][pred_cols]
    test_y_log = punt_received_df[punt_received_df['playkey'].isin(test_plays['playkey'])]['fairCatch']
    #train_x_log, test_x_log, train_y_log, test_y_log = train_test_split(X_log, Y_log, test_size=0.2) # edit this so that different plays aren't split up
    returner_features_at_catch_reg = xgb.XGBClassifier(n_estimators=20, tree_method="exact", eval_metric='logloss')
    returner_features_at_catch_reg.fit(train_x_log, train_y_log, eval_set=[(test_x_log, test_y_log)])
    preds = returner_features_at_catch_reg.predict(test_x_log)
    print(metrics.f1_score(test_y_log, preds))
    #xgb.plot_importance(returner_features_at_catch_reg)
    #plt.rcParams['figure.figsize'] = [5, 5]
    #plt.show()

    at_start_df = merged_df[merged_df['secondsSinceSnap'] == 0]
    pd.set_option('display.max_rows', 51)
    #print(at_start_df[['playkey', 'frameId', 'nflId_one', 'nflId_two', 'nflId_returner', 'secondsSinceSnap']].head(50))
    pd.set_option('display.max_rows', 10)
    pred_cols = ['x_one', 'y_one', 'o_one', 'dir_one', 'x_two', 'y_two', 'o_two', 'dir_two', 'x_returner', 'y_returner', 'dir_returner', 'o_returner', 'secondsSinceSnap']
    train_x_log = at_start_df[at_start_df['playkey'].isin(train_plays['playkey'])][pred_cols]
    train_y_log = at_start_df[at_start_df['playkey'].isin(train_plays['playkey'])]['fairCatch']
    test_x_log = at_start_df[at_start_df['playkey'].isin(test_plays['playkey'])][pred_cols]
    test_y_log = at_start_df[at_start_df['playkey'].isin(test_plays['playkey'])]['fairCatch']
    #train_x_log, test_x_log, train_y_log, test_y_log = train_test_split(X_log, Y_log, test_size=0.2) # edit this so that different plays aren't split up
    all_features_at_catch_reg = xgb.XGBClassifier(n_estimators=20, tree_method="hist", eval_metric='logloss')
    all_features_at_catch_reg.fit(train_x_log, train_y_log, eval_set=[(test_x_log, test_y_log)])
    preds = all_features_at_catch_reg.predict(test_x_log)
    print(metrics.f1_score(test_y_log, preds))
    xgb.plot_importance(all_features_at_catch_reg)
    plt.rcParams['figure.figsize'] = [5, 5]
    #plt.show()

    merged_df['kickReturnYardage'] = merged_df['kickReturnYardage'].fillna(0)
    pred_cols = ['distance_one', 'distance_two', 'totalTime', 'kickLength', 'x_one', 'y_one', 's_one', 'o_one', 'dir_one', 'x_two', 'y_two', 's_two', 'o_two', 'dir_two', 'x_returner', 'y_returner', 's_returner', 'a_returner', 'dir_returner', 'o_returner', 'secondsSinceSnap']
    train_x_lin = merged_df[merged_df['playkey'].isin(train_plays['playkey'])][pred_cols]
    print(train_x_log.head(20))
    train_y_lin = merged_df[merged_df['playkey'].isin(train_plays['playkey'])]['kickReturnYardage']
    test_x_lin = merged_df[merged_df['playkey'].isin(test_plays['playkey'])][pred_cols]
    test_y_lin = merged_df[merged_df['playkey'].isin(test_plays['playkey'])]['kickReturnYardage']
    yards_reg = xgb.XGBRegressor(n_estimators=20, tree_method="hist", eval_metric=metrics.mean_squared_error)
    yards_reg.fit(train_x_lin, train_y_lin, eval_set=[(test_x_lin, test_y_lin)])
    print(yards_reg.score(test_x_lin, test_y_lin))
    return
    #X = df[['distance', 'hangTime', 'kickLength', 'returner_movement_forward', 'a_returner']].values
    X_lin = comb[[when, 'hangTime', 'kickLength', 'a_returner'+app, 'returner_movement_forward'+app,]].values
    Y_lin = comb['kickReturnYardage'].values
    train_x_lin, test_x_lin, train_y_lin, test_y_lin = train_test_split(X_lin, Y_lin, test_size=0.2)
    reg = xgb.XGBRegressor(n_estimators=20, tree_method="hist", eval_metric=metrics.mean_absolute_error)
    reg.fit(train_x_lin, train_y_lin, eval_set=[(test_x_lin, test_y_lin)])
    print(reg.score(test_x_lin, test_y_lin))
    '''
def topGunners():
    starting_position_df = pd.read_pickle('Data/gunners_distances.pkl')
    starting_position_df['closest_other'] = starting_position_df.apply(lambda x: sorted(x.Distances)[1], axis=1)
    starting_position_df['second_closest'] = starting_position_df.apply(lambda x: sorted(x.Distances)[2], axis=1)
    starting_position_df['double_teamed'] = (starting_position_df['second_closest'] < 5).astype(int)

    punt_received_df = pd.read_pickle("Data/processed_gunner_returner_df.pkl")
    #print(len(punt_received_df))
    #print(punt_received_df[['displayName_gunner','frameId']].head(20))
    
    comb = punt_received_df.merge(starting_position_df, left_on=['gameId', 'playId', 'nflId_gunner', 'displayName_gunner'], right_on=['gameId', 'playId', 'nflId', 'displayName'])
    pred_cols = ['operationTime_x', 'hangTime_x', 'kickLength_x', 'double_teamed', 'x_returner', 'y_returner', 's_returner', 'a_returner', 'dir_returner', 'o_returner']
    #print(comb.columns.tolist())
    #print(len(comb))
    plays = pd.read_pickle("Data/combplaydata.pkl")
    punt_df = plays[plays['specialTeamsPlayType'] == 'Punt']
    punt_df['numGunners'] = punt_df['gunners'].apply(lambda x: str(x).count(';') + 1 if x == x else 0)
    punt_df = punt_df[punt_df['numGunners'] == 2]
    comb = comb.merge(punt_df, on=['gameId', 'playId'])
    X_lin = comb[pred_cols].values
    Y_lin = comb['distance_at_catch'].values

    xgbr = xgb.XGBRegressor(booster='gblinear', tree_method='hist').fit(X_lin, Y_lin)
    print(xgbr.coef_)
    print(xgbr.intercept_)
    print(xgbr.score(X_lin, Y_lin))

    comb['y_distance'] = comb.apply(,axis=1)
    X_lin = comb[pred_cols].values
    Y_lin = comb['distance_at_catch'].values

    xgbr = xgb.XGBRegressor(booster='gblinear', tree_method='hist').fit(X_lin, Y_lin)
    print(xgbr.coef_)
    print(xgbr.intercept_)
    print(xgbr.score(X_lin, Y_lin))


    log_reg = LogisticRegression(random_state=0).fit(X_log, Y_log)
    print(log_reg.score(X_log, Y_log))
    print(metrics.confusion_matrix(log_reg.predict(X_log), Y_log))

    test = sm.add_constant(X_log)
    model = sm.OLS(Y_log, test).fit() 
    #print("")
    print(model.summary())
    ## Top gunners
    preds = reg.predict(X_lin)
    comb['preds'] = pd.Series(preds)
    if when == "distance_at_catch":
        comb['distanceCloser'] = comb.apply(lambda x: x.preds - x.distance_at_catch, axis=1)
    elif when == "distance_5back":
        comb['distanceCloser'] = comb.apply(lambda x: x.preds - x.distance_5back, axis=1)
    fc_prob = log_reg.predict_proba(X_log)
    comb['fairCatchPredictedProb'] = pd.Series(fc_prob[:, 0])
    X_log = comb[['preds', 'totalTime', 'kickLength', 'a_returner'+app, 'returner_movement_forward'+app, 'returner_distance_from_endzone_less_than10'+app]].values
    pred_prob_based_on_pred_distance = log_reg.predict_proba(X_log)
    prob_effect = fc_prob - pred_prob_based_on_pred_distance
    #return
    comb['probEffect'] = pd.Series(prob_effect[:, 1])
    df = comb.groupby(by=['displayName_gunner'])[['hangTime', 'double_teamed', 'distanceCloser', when, 'preds', 'kickReturnYardage', 'fairCatchPredictedProb', 'probEffect']].agg(['mean', 'count']).reset_index()

    df = df.sort_values(by=[('probEffect', 'mean')], ascending=False)
    df = df[df[(when, 'count')] > 30]
    df = df[[(when, 'count'), ('distanceCloser', 'mean'), ('displayName_gunner',''), (when, 'mean'), ('double_teamed', 'mean'), 
    ('hangTime', 'mean'), ('kickReturnYardage', 'mean'), ('fairCatchPredictedProb', 'mean'), ('probEffect', 'mean')]]
    print(len(df))
    print(df.head(20))
    df.to_csv('topGunners.csv')

#xgbModelPrep()
#xgbmodelPrep2()
#main()
#main()
topGunners()
#main(sys.argv[1])
#puntRegressionAnalysis()
#preprocessGunnersReturnersdf()