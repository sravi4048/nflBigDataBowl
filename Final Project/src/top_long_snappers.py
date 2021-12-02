import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import LinearRegression, LogisticRegression
import statsmodels.api as sm
from sklearn import metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
from IPython.display import display

def xgb_f1(y: np.ndarray, t: np.ndarray, threshold=0.5):
    #t = t.get_label()
    t = (t > threshold).astype(int)
    y_bin = (y > threshold).astype(int) # works for both type(y) == <class 'numpy.ndarray'> and type(y) == <class 'pandas.core.series.Series'>
    return 1 - f1_score(t,y_bin)

def regress():
    df = pd.read_pickle("Data/long_snapper_data.pkl")
    df = df[df['specialTeamsPlayType'] == 'Punt']
    df['goodSnap'] = (df['snapDetail'] == 'OK').astype(int)
    df = df.dropna(subset=['snapTime', 'snapDetail'])
    df = pd.get_dummies(df, columns=['snapDetail'])
    df = df[df['snapTime'] > 0.6]
    df = df[df['snapTime'] < 1.1]
    df['penaltyYards'] = df['penaltyYards'].fillna(0)

    #print(df.groupby(by=['specialTeamsResult'])['playResult'].mean())
    #print(df.columns.tolist())
    df['badOutcome'] = (df['specialTeamsResult'].isin(['Blocked Punt', 'Non-Special Teams Result'])).astype(int)
    #print(df['badOutcome'].values.sum())
    #print(df['badOutcome'].values.sum()/len(df['badOutcome']))
    #print(len(df))
    print(confusion_matrix(df['badOutcome'], df['goodSnap'], labels=[1,0]))
    #print(classification_report(df['badOutcome'], df['goodSnap']))
    sm = SMOTE(sampling_strategy='minority', random_state=0)
    df['snap_left'] = df['snapDetail_<']
    df['snap_right'] = df['snapDetail_>']
    df['snap_High'] = df['snapDetail_H']
    df['snap_Low'] = df['snapDetail_L']
    df['snap_OK'] = df['snapDetail_OK']

    X = df[['snap_left', 'snap_right', 'snap_High', 'snap_Low', 'snap_OK', 'snapTime']]
    Y = df['badOutcome']
    train_x_log, test_x_log, train_y_log, test_y_log = train_test_split(X, Y, test_size=0.2)
    oversampled_train_x, oversampled_train_y = sm.fit_resample(train_x_log, train_y_log)
    print(len(train_x_log), len(oversampled_train_x))
    reg = LogisticRegression(random_state=0).fit(oversampled_train_x, oversampled_train_y)
    print(reg.coef_)
    print(reg.intercept_)
    print(reg.score(X, Y))
    preds = reg.predict(test_x_log)
    print(metrics.f1_score(test_y_log, preds))
    print(confusion_matrix(Y, reg.predict(X), labels=[1,0]))
    print(classification_report(Y, reg.predict(X)))
    #return
    xgbReg = xgb.XGBClassifier(n_estimators=100, tree_method="hist", learning_rate=0.3, eval_metric=xgb_f1)
    xgbReg.fit(oversampled_train_x, oversampled_train_y, eval_set=[(test_x_log, test_y_log)])
    preds = xgbReg.predict(test_x_log)
    print(metrics.f1_score(test_y_log, preds))
    print(confusion_matrix(Y, xgbReg.predict(X), labels=[1,0]))
    print(classification_report(Y, xgbReg.predict(X)))
    xgb.plot_importance(xgbReg)
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.show()
    xgb.plot_tree(xgbReg,num_trees=49)
    plt.rcParams['figure.figsize'] = [50, 10]
    plt.show()
    return
    #df = df.dropna(subset=['kickReturnYardage'])
    df = df.dropna(subset=['kickLength', 'goodSnap', 'playResult'])
    X = df[['snapTime', 'goodSnap', 'kickLength']].values
    Y = df['playResult'].values
    reg = LinearRegression().fit(X, Y)
    print(reg.coef_)
    print(reg.intercept_)
    print(reg.score(X, Y))
    plt.clf()
    plt.scatter(X[:, 0], reg.predict(X) - Y)
    plt.title("Residuals plot")
    plt.xlabel("X")
    plt.ylabel("Residual")
    #plt.show()
    plt.clf()
    plt.scatter(X[:, 0], Y)
    #plt.show()
    plt.clf()
    plt.scatter(X[:, 0], reg.predict(X))
    #plt.show()

    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit() 
    print("")
    print(model.params)
    print(model.tvalues)
    print(model.bse)
    print(model.pvalues)

def bestLongSnappers():
    df = pd.read_pickle("Data/long_snapper_data.pkl")
    df['snapDetail']


def topLongSnappers():
    df = pd.read_pickle("Data/long_snapper_data.pkl")
    snap_detail_results = df[['displayName', 'snapDetail']]
    table = snap_detail_results.pivot_table(index=['displayName'], columns=['snapDetail'], aggfunc=len, fill_value=0)
    #print(table.head(50))
    #snap_time_df = df[['displayName', 'nflId', 'specialTeamsResult', 'snapTime']]
    table['total'] = table['<'] + table['>'] + table['H'] + table['L'] + table['OK']
    cols = ['<', '>', 'H', 'L', 'OK']
    for col in cols:
        table[col] = 100 * (table[col] / table['total'])
    table = table.sort_values(by=['OK', 'total', '<', '>', 'H'], ascending=False)
    print(table)
    #table.to_pickle("Data/longsnapaccuracy.pkl")
    #print(len(count))

def main():
    df = pd.read_pickle("Data/long_snapper_data.pkl")
    df = df[['gameId', 'playId', 'displayName', 'nflId', 'specialTeamsPlayType', 'specialTeamsResult', 'snapTime', 'snapDetail', 'operationTime', 'kickLength', 'kickReturnYardage', 'playResult']]
    snap_time = df[['displayName', 'nflId', 'snapTime']]
    snap_time_means = snap_time.groupby(['displayName'])['snapTime'].agg(['mean', 'std', 'count'])
    snap_time_means = snap_time_means.sort_values(by=['mean', 'std', 'count'], ascending=True)
    #snap_time_means['std'] = snap_time_means.apply(lambda x: x.var ** 0.5, axis=1)
    print(snap_time_means[['count', 'mean', 'std']])
    return
    df = pd.read_pickle("Data/long_snapper_data.pkl")
    snap_detail_results = df[['displayName', 'snapDetail']]
    table = snap_detail_results.pivot_table(index=['displayName'], columns=['snapDetail'], aggfunc=len, fill_value=0)
    table['total'] = table['<'] + table['>'] + table['H'] + table['L'] + table['OK']
    cols = ['<', '>', 'H', 'L', 'OK']
    for col in cols:
        table[col] = 100 * (table[col] / table['total'])
    #table = table.sort_values(by=['OK', 'total', '<', '>', 'H'], ascending=False)
    table = table.merge(snap_time_means, on=['displayName'])
    table['combinedMetric'] = (100 *table['mean']) / table['OK']
    table = table.sort_values(by=['combinedMetric'], ascending=True)[table['total'] > 30]
    cm = sns.light_palette("green", as_cmap=True)
    table = table.rename(columns={'mean': 'Mean Snap Time', 'OK': 'Snap Accuracy', 'combinedMetric': 'Combined Metric'})
    print(table.columns.tolist())
    print(table[['Mean Snap Time', 'Snap Accuracy', 'Combined Metric']])
    #style = table[['Mean Snap Time', 'Snap Accuracy']].style.background_gradient(cmap=cm).format(precision=2)
    #display(style)
    table[['Mean Snap Time', 'Snap Accuracy', 'Combined Metric']].to_csv('TopLongSnappers.csv')

main()