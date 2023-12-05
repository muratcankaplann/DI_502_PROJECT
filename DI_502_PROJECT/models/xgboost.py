import pandas as pd
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df_=pd.read_csv("../data/df.csv",delimiter=";")
df=df_.copy()

def generate_lagged_features(dataframe, n_step,n_next):
    """
    Generate lagged features for the demand based on the n_step value and store the next 24 demand values as an array.

    Args:
    - df (pd.DataFrame): Original dataframe
    - n_step (int): Number of hours to consider for previous demand

    Returns:
    - df_new (pd.DataFrame): Dataframe with new features
    """
    data=dataframe.copy()
    df_=pd.DataFrame()
    df_['demand_lag_1'] = data['total_demand'].shift(1)
    # Creating lag features for demand
    for i in range(2, n_step + 1):
        df_dummy=pd.Series(data['total_demand'].shift(i), name=f'demand_lag_{i}')
        df_ = pd.concat([df_, df_dummy], axis=1)
    
    # Dropping rows with NaN values
    df_new = pd.concat([data, df_], axis=1)
    df_new=df_new.dropna()


    # Resetting index
    df_new = df_new.reset_index(drop=True)
    # Store the next 24 demand values as an array for the rows that will remain in the DataFrame
    next_demand_values = []
    # Calculate the next 24 demand values for each row
    for i in range(len(df_new) - n_next):
        next_demand_values.append(df_new['total_demand'].tolist()[i+1:i+n_next+1])
    print(np.array(next_demand_values).shape)
    df_new = df_new.iloc[:-n_next]
    # Add the next_24_demand values as a new column
    df_new['total_demand_'] = next_demand_values

    # Reshape the data for regression modeling
    df_test_ = pd.DataFrame(df_new['total_demand_'].tolist(), columns=[f'demand_{i+1}_hour' for i in range(n_next)])
    df_new = df_new.drop('total_demand_', axis=1)

    # Resetting index
    df_new = df_new.reset_index(drop=True)

    
    return df_new,df_test_

def generate_past_demand(dataframe, n_step):
    """
    Generate lagged features for the demand based on the n_step value and store the next 24 demand values as an array.

    Args:
    - df (pd.DataFrame): Original dataframe
    - n_step (int): Number of hours to consider for previous demand

    Returns:
    - df_new (pd.DataFrame): Dataframe with new features
    """
    data=dataframe.copy()
    if isinstance(data, pd.Series):
        data=data.to_frame().T
    df_=pd.DataFrame()
    df_['demand_lag_1'] = data['total_demand'].shift(1)
    # Creating lag features for demand
    for i in range(2, n_step + 1):
        df_dummy=pd.Series(data['total_demand'].shift(i), name=f'demand_lag_{i}')
        df_ = pd.concat([df_, df_dummy], axis=1)
    
    # Dropping rows with NaN values
    df_new = pd.concat([data, df_], axis=1)
    df_new=df_new.dropna()


    # Resetting index
    df_new = df_new.reset_index(drop=True)

    return df_new


def minmax_scale_dataframe(data):
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return df_scaled
n_prev=168
n_next=8
df_X,df_y = generate_lagged_features(df, n_step = n_prev,n_next=n_next)
from sklearn.linear_model import LinearRegression,ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
split_val_index = int(len(df_X) * 2 / 3)
split_test_index = int(len(df_X) * 5 / 6)
# Separate independent and dependent variables for training set
X_train = df_X.loc[:split_val_index].values
y_train = df_y.loc[:split_val_index].values

# Separate independent and dependent variables for validation set
X_val = df_X.loc[split_val_index:split_test_index].values
y_val = df_y.loc[split_val_index:split_test_index].values

X_test = df_X.loc[split_test_index:].values
y_test = df_y.loc[split_test_index:].values


scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train=scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
X_val = scaler_X.transform(X_val)
model = xgb.XGBRegressor(booster='gbtree',objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.3,
                        max_depth = 5,sampling_method="gradient_based", alpha = 10, n_estimators = 500,early_stopping_rounds=5,device="cuda",grow_policy="lossguide")

model.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_test, y_test)],verbose=1)


from sklearn.metrics import mean_squared_error
y_pred_test = model.predict(X_test)
y_pred_val = model.predict(X_val)
def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual) * 100, axis=0)

# Evaluate the model
mape_test = calculate_mape(y_test, y_pred_test)
mape_val = calculate_mape(y_val, y_pred_val)
mse_test = mean_squared_error(y_test, y_pred_test)
mse_val = mean_squared_error(y_val, y_pred_val)
print(f"{n_next} hours prediction Mean Absolute Percentage Error for Test Dataset:", mape_test.mean())
print(f"{n_next} hours prediction Mean Absolute Percentage Error for Validation Dataset:", mape_val.mean())
print(f"{n_next} hours prediction Mean Squared Error for Test Dataset:", np.sqrt(mse_test.mean()))
print(f"{n_next} hours prediction Mean Squared Error for Validation Dataset:", np.sqrt(mse_val.mean()))