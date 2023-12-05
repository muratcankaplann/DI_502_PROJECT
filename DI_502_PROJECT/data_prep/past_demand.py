import pandas
def generate_past_demand(dataframe, n_step):
    """
    Generate lagged features for the demand based on the n_step value and store the next 24 demand values as an array.

    Args:
    - dataframe (pd.DataFrame or pd.Series): Original dataframe or series containing the demand values.
    - n_step (int): Number of hours to consider for previous demand.

    Returns:
    - df_new (pd.DataFrame): Dataframe with new lagged features.
    """

    # Ensure the input is a dataframe; if it's a series, convert it to a dataframe
    data = dataframe.copy()
    if isinstance(data, pd.Series):
        data = data.to_frame().T

    # Initialize a new dataframe to store lagged features
    df_ = pd.DataFrame()

    # Create lag features for demand
    df_['demand_lag_1'] = data['total_demand'].shift(1)
    
    # Creating lag features for demand
    for i in range(2, n_step + 1):
        df_dummy = pd.Series(data['total_demand'].shift(i), name=f'demand_lag_{i}')
        df_ = pd.concat([df_, df_dummy], axis=1)

    # Combine original data and lagged features
    df_new = pd.concat([data, df_], axis=1)

    # Drop rows with NaN values
    df_new = df_new.dropna()

    # Resetting index
    df_new = df_new.reset_index(drop=True)

    return df_new