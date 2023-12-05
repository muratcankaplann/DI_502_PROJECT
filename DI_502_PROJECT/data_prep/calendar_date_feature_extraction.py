import pandas as pd
special_holidays = [
    "1-1",  # New Year's Day
    "2-1",  # New Year's Day (additional day)
    "26-1",  # Australia Day
    "13-3",  # Labour Day
    "7-4",  # Good Friday
    "8-4",  # Easter Saturday
    "10-4",  # Easter Monday
    "25-4",  # Anzac Day
    "12-6",  # King's Birthday
    "7-11",  # Melbourne Cup
    "25-12",  # Christmas Day
    "26-12"  # Boxing Day
]
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Saturday', 'Sunday']
season=["autumn","spring","winter"]

def format_date(date):
    formatted = date.strftime("%d-%m")
    day, month = formatted.split('-')
    return f"{int(day)}-{int(month)}"

def get_season(date):
    month = date.month
    if 12 <= month <= 2:
        return 'summer'
    elif 3 <= month <= 5:
        return 'autumn'
    elif 6 <= month <= 8:
        return 'winter'
    elif 9 <= month <= 11:
        return 'spring'
    
def feature_extraction_pipeline(data):
    df_=data.copy()
    if isinstance(df_, pd.Series):
        df_=df_.to_frame().T
        
    # Ensure valid_start column is of type datetime
    df_['valid_start'] = pd.to_datetime(df_['valid_start'], format='%d.%m.%Y %H:%M')

    # Extract the day of the week
    df_['day_of_week'] = df_['valid_start'].dt.day_name()

    # One-hot encode the day_of_week column 
    df_encoded = pd.get_dummies(df_, columns=['day_of_week'],dtype=int)
    for day in days_of_week:
        if f'day_of_week_{day}' not in df_encoded.columns:
            df_encoded[f'day_of_week_{day}'] = 0
    df_encoded['week_of_year'] = df_encoded['valid_start'].dt.isocalendar().week
    df_encoded['is_special_day'] = df_encoded['valid_start'].apply(lambda x: 1 if format_date(x) in special_holidays else 0)

    # Add a 'season' column to the dataframe
    df_encoded['season'] = df_encoded['valid_start'].apply(get_season)

    # One-hot encode the 'season' column
    df_encoded = pd.get_dummies(df_encoded, columns=['season'], prefix='season',dtype=int)
    for day in season:
        if f'season_{day}' not in df_encoded.columns:
            df_encoded[f'season_{day}'] = 0
    df_encoded['hour_of_day'] = df_encoded['valid_start'].dt.hour

    if 'day_of_week_Friday' in df_encoded.columns:
        df_encoded = df_encoded.drop('day_of_week_Friday', axis=1)
    if 'season_summer' in df_encoded.columns:
        df_encoded = df_encoded.drop('season_summer', axis=1)
    return df_encoded