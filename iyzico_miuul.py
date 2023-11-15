############### Import Library and Settings ###################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import re
import lightgbm as lgb
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
warnings.filterwarnings('ignore')


############### Get Data ###################
data = pd.read_csv("datasets/iyzico_data.csv", parse_dates = ["transaction_date"])
data = data[data.columns[1:]]
data.head()
data.info()


############### EDA - 1 ###################
## Mim and Max date of data
min_date, max_date = data["transaction_date"].min(), data["transaction_date"].max()
print(f"minimum date in data: {min_date}\nmaksimum date in data: {max_date}")

## Unique business number
data["merchant_id"].unique()

## Number of businesses appearing in the data
data["merchant_id"].value_counts()

## Total number of transactions and amount paid
data.groupby("merchant_id").agg({"Total_Paid": "sum",
                                 "Total_Transaction": "sum"}).reset_index()

## Annual graphs of businesses
for id in data.merchant_id.unique():
    plt.figure(figsize = (15, 15))
    plt.subplot(2, 1, 1, title = str(id) + ' 2018-2019 Transaction Count')
    data[(data.merchant_id == id) & (data.transaction_date >= "2018-01-01") & (data.transaction_date < "2019-01-01")][
        "Total_Transaction"].plot()
    plt.subplot(2, 1, 2, title = str(id) + ' 2019-2020 Transaction Count')
    data[(data.merchant_id == id) & (data.transaction_date >= "2019-01-01") & (data.transaction_date < "2020-01-01")][
        "Total_Transaction"].plot()
    plt.show()


############### Feature Engineering ###################
### 1- Date Feature
def create_date_features(df, date_column):
    df['month'] = df[date_column].dt.month
    df['day_of_month'] = df[date_column].dt.day
    df['day_of_year'] = df[date_column].dt.dayofyear
    df['week_of_year'] = df[date_column].dt.isocalendar().week
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['year'] = df[date_column].dt.year
    df["is_wknd"] = df[date_column].dt.weekday // 4
    df['is_month_start'] = df[date_column].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_column].dt.is_month_end.astype(int)
    df['quarter'] = df[date_column].dt.quarter
    df['is_quarter_start'] = df[date_column].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df[date_column].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df[date_column].dt.is_year_start.astype(int)
    df['is_year_end'] = df[date_column].dt.is_year_end.astype(int)
    return df


data = create_date_features(data, "transaction_date")
data['week_of_year'] = data['week_of_year'].astype('int')
data.head()
data.info()

data.groupby(["merchant_id", "year", "month"]).agg({"Total_Transaction": ["sum", "mean", "median"]}).reset_index()
data.groupby(["merchant_id", "year", "month"]).agg({"Total_Paid": ["sum", "mean", "median"]}).reset_index()


### 2- Lag Feature
def random_noise(dataframe):
    return np.random.normal(scale = 1.6, size = (len(dataframe),))


def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["merchant_id"])['Total_Transaction']. \
                                                 transform(lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe


lags = [91, 92, 93, 94, 95, 96, 97, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185,
        186, 187, 188, 189, 190, 350, 351, 352, 352, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366,
        367, 368, 369, 370, 538, 539, 540, 541, 542, 718, 719, 720, 721, 722]
data = lag_features(data, lags)


### 3- Roll Mean Feature

def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = \
            dataframe.groupby("merchant_id")['Total_Transaction'].transform(
                lambda x: x.shift(1).rolling(window = window, min_periods = 10,
                                             win_type = "triang").mean()) + random_noise(
                dataframe)
    return dataframe


roll = [91, 92, 178, 179, 180, 181, 182, 270, 271, 359, 360, 361, 449, 450, 451, 539, 540, 541, 629, 630, 631, 720]
data = roll_mean_features(data, roll)


### 4- Exponentially Weighted Mean Features
def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby("merchant_id")['Total_Transaction'].transform(
                    lambda x: x.shift(lag).ewm(alpha = alpha).mean())
    return dataframe


alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 92, 178, 179, 180, 181, 182, 270, 271, 359, 360, 361, 449, 450, 451, 539, 540, 541, 629, 630, 631, 720]

data = ewm_features(data, alphas, lags)

### 5- Special Days Feature
data["is_black_friday"] = 0
data.loc[data["transaction_date"].isin(["2018-11-22", "2018-11-23",
                                        "2019-11-29", "2019-11-30",
                                        "2020-11-26", "2020-11-27"]), "is_black_friday"] = 1

data["is_summer_solstice"] = 0
data.loc[data["transaction_date"].isin(["2018-06-19", "2018-06-20", "2018-06-21", "2018-06-22",
                                        "2019-06-19", "2019-06-20", "2019-06-21", "2019-06-22",
                                        "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22"]
                                       ), "is_summer_solstice"] = 1

data["sacrifices_day"] = 0
data.loc[data["transaction_date"].isin(["2018-08-18", "2018-08-19", "2018-08-20",
                                        "2019-08-08", "2019-08-09", "2019-08-10",
                                        "2020-07-28", "2020-07-29", "2020-07-30"]
                                       ), "is_summer_solstice"] = 1
data["ramadan_day"] = 0
data.loc[data["transaction_date"].isin(["2018-06-11", "2018-06-11", "2018-06-12",
                                        "2019-06-01", "2019-06-02", "2019-06-03",
                                        "2020-05-21", "2020-05-22", "2020-05-23"]
                                       ), "is_summer_solstice"] = 1


###########  One-Hot Encoding ###########
data.head()
df = pd.get_dummies(data, columns=['merchant_id', 'day_of_week', 'day_of_month', 'month', 'year'], dtype='int')
df['Total_Transaction'] = np.log1p(df["Total_Transaction"].values)
df.info()


########### Custome Cost Function #########
def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False


########### Train and Validation Sets Split ###########
df = df.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
# Dataframe col names are editted.
df.head()

train = df.loc[(df["transaction_date"] < "2020-10-01"), :]

val = df.loc[(df["transaction_date"] >= "2020-10-01"), :]

cols = [col for col in train.columns if col not in ['transaction_date', "Total_Transaction","Total_Paid" ]]

Y_train = train['Total_Transaction']
X_train = train[cols]

Y_val = val['Total_Transaction']
X_val = val[cols]


########################
# LightGBM Model
########################
# LightGBM parameters
lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.01,
              'feature_fraction': 0.7,
              'max_depth': 10,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,
              'nthread': -1}

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))
#20.836761758092802


############ Feature Importance ######
def plot_lgb_importances(model, plot=False, num=10):

    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))


plot_lgb_importances(model, num=30, plot=True)

lgb.plot_importance(model, max_num_features=20, figsize=(10, 10), importance_type="gain")
plt.show()