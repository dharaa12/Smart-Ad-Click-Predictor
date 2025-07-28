
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from avazu_dataset_handler import load_avazu_gz
import matplotlib.pyplot as plt
import seaborn as sns

class AvazuEDA:
    def __init__(self, train_df, test_df,  main_df = 'train'):

        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        if main_df == 'train':
            self.df = self.train_df.copy()
        else:
            self.df = self.test_df.copy()
        print(f"Train dataset shape: {self.train_df.shape}")
        print(f"Test dataset shape: {self.test_df.shape}")


    def plot_target_distribution(self):

        plt.figure(figsize=(12, 4))


        plt.subplot(1, 2, 1)
        click_counts = self.df['click'].value_counts()
        plt.pie(click_counts.values, labels=['No Click (0)', 'Click (1)'],
                autopct='%1.2f%%', startangle=90)
        plt.title('Click Distribution')


        plt.subplot(1, 2, 2)


        self.df['hour_only'] = self.df['hour'].astype(str).str[-2:].astype(int)
        hourly_ctr = self.df.groupby('hour_only')['click'].mean()
        print(hourly_ctr.index, hourly_ctr.values )
        plt.plot(hourly_ctr.index, hourly_ctr.values, marker='o')
        plt.title('Click Rate by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Click Rate')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        print(f"Overall CTR: {self.df['click'].mean():.4f}")



    def plot_categorical_features(self):

        cols_to_plot = [col for col in self.df.columns ]
        if len(cols_to_plot) > 0:
            n_rows = 5
            n_cols = 5
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
            axes = axes.flatten()

            for i, col in enumerate(cols_to_plot):
                ax = axes[i]

                top_cats = self.df[col].value_counts().head(15)
                ax.bar(range(len(top_cats)), top_cats.values)
                ax.set_title(f'{col} Distribution (Top 10)')
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
                if len(top_cats) <= 10:
                    ax.set_xticks(range(len(top_cats)))
                    ax.set_xticklabels(top_cats.index, rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

    def plot_feature_ctr_analysis(self):

        cols_to_plot = [col for col in self.df.columns
                    if col not in ['id', 'click']]
        n_rows = 4
        n_cols = 6
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()
        for i, col in enumerate(cols_to_plot):
            ax = axes[i]
            ctr_by_cat = self.df.groupby(col)['click'].agg(['mean', 'count']).reset_index()
            ctr_by_cat = ctr_by_cat.nlargest(10, 'count')
            if len(ctr_by_cat) > 0:
                bars = ax.bar(range(len(ctr_by_cat)), ctr_by_cat['mean'])
                ax.set_title(f'CTR by {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Click Rate')
                ax.set_xticks(range(len(ctr_by_cat)))
                ax.set_xticklabels(ctr_by_cat[col], rotation=45, ha='right')

                # Add value labels on bars
                for bar, value in zip(bars, ctr_by_cat['mean']):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                            f'{value:.4f}', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.show()

    def plot_device_analysis(self):
        device_cols = [col for col in self.df.columns if 'device' in col.lower()]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, col in enumerate(device_cols):
            ax = axes[i]
            device_stats = self.df.groupby(col).agg({ 'click': ['mean', 'count']}).round(4)
            device_stats.columns = ['CTR', 'Count']
            device_stats = device_stats.sort_values('Count', ascending=False).head(10)
            if len(device_stats) > 0:
                ax2 = ax.twinx()
                bars1 = ax.bar(range(len(device_stats)), device_stats['Count'],
                               alpha=0.7, color='lightblue', label='Count')
                line1 = ax2.plot(range(len(device_stats)), device_stats['CTR'], label='CTR')
                ax.set_xlabel(col)
                ax.set_ylabel('Count', color='blue')
                ax2.set_ylabel('CTR', color='pink')
                ax.set_title(f'{col} Distribution and CTR')
                ax.set_xticks(range(len(device_stats)))
                ax.set_xticklabels(device_stats.index, rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


    def plot_correlation_analysis_numerical_values(self):
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_plot = [col for col in self.df.columns
                    if col not in ['id', 'click']]
        plt.figure(figsize=(12, 10))
        corr_matrix = self.df[num_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
                    center=0, square=True, fmt='.3f')
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.show()


    def plot_correlation_analysis(self):
        all_cols = [col for col in self.df.columns if col not in ['id', 'hour']]
        processed_df =self.df[all_cols].copy()

        categorical_cols = []
        numerical_cols = []

        for col in processed_df.columns:
            if processed_df[col].dtype == 'object' or processed_df[col].nunique() < 50:
                if col != 'click':
                    categorical_cols.append(col)
                    le = LabelEncoder()
                    processed_df[col] = processed_df[col].fillna('Unknown')
                    processed_df[col] = le.fit_transform(processed_df[col].astype(str))
                else:
                    numerical_cols.append(col)
            else:
                numerical_cols.append(col)


        corr_matrix = processed_df.corr()
        fig = plt.figure(figsize=(20, 15))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
                    center=0, square=True, fmt='.3f')
        plt.title('Full Feature Correlation Matrix\n(Categorical features label-encoded)')
        plt.tight_layout()
        plt.show()





    def datset_analysis(self):

        # tested

        self.plot_target_distribution()

        # tested

        self.plot_categorical_features()


        # tested

        self.plot_feature_ctr_analysis()

        # tested could be applied to other keywords
        self.plot_device_analysis()

        #
        self.plot_correlation_analysis()

        self.plot_correlation_analysis_numerical_values()



def dataset_inspection(df):
    df_clean = df.copy()
    print(df.columns)

    print(f"Dataset shape: {df_clean.shape}")
    print(f"Columns: {list(df_clean.columns)}")


    df.rename(columns={
        'banner_pos': 'banner_position',
        'device_type': 'device_type',
        'device_conn_type': 'device_connection_type',
        # Optional: clarify anonymized columns
        'C1': 'cat_C1',
        'C14': 'cat_C14', 'C15': 'cat_C15', 'C16': 'cat_C16',
        'C17': 'cat_C17', 'C18': 'cat_C18', 'C19': 'cat_C19',
        'C20': 'cat_C20', 'C21': 'cat_C21'
    }, inplace=True)

    print(df.info())
    print(df.head())

    print(df.head(10))

    for dt in df['device_type'].unique():
        print(f"Device Type: {dt}")
        print(df[df['device_type'] == dt]['device_model'].value_counts().head(10))
        print("\n")

    df.isnull().sum()

    df['hour'] = pd.to_datetime(df['hour'], format='%y%m%d%H')
    df['hour_of_day'] = df['hour'].dt.hour
    df['day_of_week'] = df['hour'].dt.dayofweek

    print(df[['hour', 'hour_of_day', 'day_of_week']].head(10))

    ctr_by_banner = df.groupby('banner_position')['click'].agg(['sum', 'count']).reset_index()
    ctr_by_banner['CTR (%)'] = (ctr_by_banner['sum'] / ctr_by_banner['count']) * 100
    ctr_by_banner.rename(columns={'sum': 'clicks', 'count': 'impressions'}, inplace=True)

    print(ctr_by_banner)


    # CTR by Device Type
    ctr_by_device = df.groupby('device_type')['click'].agg(['sum', 'count']).reset_index()
    ctr_by_device['CTR (%)'] = (ctr_by_device['sum'] / ctr_by_device['count']) * 100
    ctr_by_device.rename(columns={'sum': 'clicks', 'count': 'impressions'}, inplace=True)
    print("CTR by Device Type:\n", ctr_by_device)

    # CTR by hour of day
    ctr_by_hour = df.groupby('hour_of_day')['click'].agg(['sum', 'count']).reset_index()
    ctr_by_hour['CTR (%)'] = (ctr_by_hour['sum'] / ctr_by_hour['count']) * 100
    ctr_by_hour.rename(columns={'sum': 'clicks', 'count': 'impressions'}, inplace=True)
    print("CTR by Hour of Day:\n", ctr_by_hour)




    ctr_by_hour_day = df.groupby(['day_of_week', 'hour_of_day'])['click'].mean().unstack()

    print("Unique hour_of_day values:", sorted(df['hour_of_day'].unique()))
    print("Unique day_of_week values:", sorted(df['day_of_week'].unique()))




if __name__ == "__main__":

    train_df, test_df = load_avazu_gz('./avazu_data', sample_size=10000)
    analyzer = AvazuEDA(train_df, test_df)
    analyzer.datset_analysis()
    dataset_inspection(train_df)