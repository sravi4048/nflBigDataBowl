import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

def main():
    pd.options.mode.chained_assignment = None
    scouting = pd.read_pickle("Data/scoutingdata.pkl")
    plays = pd.read_pickle("Data/plays.pkl")
    kickoff_plays = scouting.merge(plays, on=['gameId', 'playId'])
    kickoff_plays = kickoff_plays[kickoff_plays['specialTeamsPlayType'] == 'Kickoff']
    cols = ['kickLength', 'kickType', 'hangTime', 'kickDirectionIntended', 'kickDirectionActual', 'returnDirectionActual', 'returnDirectionIntended', 'missedTackler', 
    'kickoffReturnFormation', 'playResult', 'kickReturnYardage']
    kickoff_plays = kickoff_plays[cols]
    print(kickoff_plays.head(20))
    kickoff_plays['kickDirectionMatches'] = (kickoff_plays['kickDirectionIntended'] == kickoff_plays['kickDirectionActual']).astype(int)
    kickoff_plays['returnDirectionMatches'] = (kickoff_plays['returnDirectionIntended'] == kickoff_plays['returnDirectionActual']).astype(int)
    kickoff_plays['numMissedTackles'] = kickoff_plays['missedTackler'].apply(lambda x: str(x).count(';') + 1 if x == x else 0)
    mask = (kickoff_plays['kickDirectionIntended'] != kickoff_plays['kickDirectionIntended'])
    kickoff_plays['kickDirectionMatches'][mask] = 1
    mask = (kickoff_plays['returnDirectionIntended'] != kickoff_plays['returnDirectionIntended'])
    kickoff_plays['returnDirectionMatches'][mask] = 1
    #print(kickoff_plays.head(20))
    kickoff_plays = kickoff_plays[kickoff_plays['kickType'] != 'K']
    kickoff_plays = pd.get_dummies(kickoff_plays, prefix='kickType', columns=['kickType'])
    kickoff_plays = pd.get_dummies(kickoff_plays, prefix='formation', columns=['kickoffReturnFormation'])
    #print(kickoff_plays.columns.tolist())
    for col in kickoff_plays.columns.tolist():
        if col.startswith('kickType'):
            for col2 in kickoff_plays.columns.tolist():
                if col2.startswith('formation'):
                    kickoff_plays[col + '_' + col2] = kickoff_plays[col] * kickoff_plays[col2]
    kickoff_plays = kickoff_plays.loc[:, (kickoff_plays != 0).any(axis=0)]
    #print(kickoff_plays.columns.tolist())
    pred_cols = ['kickLength', 'hangTime', 'kickDirectionMatches', 'returnDirectionMatches',
    'formation_10-0-0', 'formation_5-0-4', 'formation_5-3-2', 'formation_6-0-3', 'formation_6-0-4', 'formation_6-2-2', 
    'formation_7-0-3', 'formation_7-1-2', 'formation_8-0-1', 'formation_8-0-2', 'formation_8-0-3', 'formation_8-1-0', 
    'formation_8-1-1', 'formation_9-0-1', 'formation_9-1-0']
    for col in kickoff_plays.columns.tolist():
        if col.startswith('kickType'):
            pred_cols.append(col)
    #print(kickoff_plays.columns.tolist())
    kickoff_plays = kickoff_plays.dropna(subset=pred_cols)
    #print(pred_cols)
    X = kickoff_plays[pred_cols].values
    Y = kickoff_plays['playResult'].values
    reg = LinearRegression().fit(X, Y)
    #print(reg.coef_)
    #print(reg.intercept_)
    #print(reg.score(X, Y))
    plt.clf()
    plt.scatter(Y, reg.predict(X) - Y)
    plt.title("Residuals plot")
    plt.xlabel("X")
    plt.ylabel("Residual")
    #plt.show()
    plt.clf()
    plt.scatter(X[:, 1], Y)
    plt.title("Scatter Hang Time vs Result")
    #plt.show()
    plt.clf()
    #plt.scatter(X[:, 0], reg.predict(X))
    #plt.show()

    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit() 
    #print("")
    #print(model.params)
    #print(model.tvalues)
    #print(model.bse)
    #print(model.pvalues)
    pred_cols.insert(0, 'intercept')
    pd.set_option('display.max_rows', None)
    params_df = pd.DataFrame([pred_cols, model.params, model.pvalues])
    params_df = params_df.transpose()
    params_df.columns = ['variables', 'coef', 'pvalues']
    params_df = params_df.sort_values(by=['pvalues'])
    params_df = params_df.dropna(subset=['pvalues'])
    #print(params_df)
    #print(len(params_df))
    #print(kickoff_plays.columns)
    view_df = kickoff_plays[['kickLength', 'hangTime', 'missedTackler', 'playResult', 'kickReturnYardage', 'formation_6-2-2']]
    #print(view_df[view_df['formation_6-2-2'] == 1])
    pd.set_option('display.max_rows', 10)
    #print(kickoff_plays['formation_6-2-2'].values.sum())
    return

main()