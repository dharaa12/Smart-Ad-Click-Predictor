import pandas as pd

def clean_data(file_path, nrows=None):
    df = pd.read_csv(file_path, compression='gzip', nrows=nrows)

    # Time features
    df['datetime'] = pd.to_datetime(df['hour'], format='%y%m%d%H')
    df['hour_of_day'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['display_hour'] = df['datetime'].dt.strftime('%I %p')
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday'])

    def get_time_bucket(hour):
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'

    df['time_bucket'] = df['hour_of_day'].apply(get_time_bucket)

    #hour not needed 
    df.drop(columns=['hour'], inplace=True)
    
    return df

if __name__ == "__main__":
    print("🚀 Starting data cleaning...")

    df = clean_data('data/train.gz')  
    print("✅ Data loaded and cleaned:", df.shape)

    pd.set_option('display.max_columns', None)
    print("🔍 Preview of cleaned data:")
    print(df.head())

    # Save cleaned CSV
    df.to_csv('data/cleaned_train.csv', index=False)
    print("📁 Cleaned data saved to: data/cleaned_train.csv")

