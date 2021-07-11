import pandas as pd
import numpy as np
import os

import setup_environment as env


"""Show parameters"""
env.set_pd_environments()
# constants
config = env.get_config()


def load_csv(show_data_overview=False):
    """load and show data"""
    df_tmp = pd.read_csv(config['files']['weatherFile'])

    """wrangle and clean data"""
    df_tmp['date'] = df_tmp['date'].astype({'date': 'datetime64[ns]'})
    # df_tmp = df_tmp.set_index(['date'])
    df_tmp = df_tmp.sort_values('date')

    # limit the time frame
    df_tmp = df_tmp[df_tmp['datetimeStr'] >= config['weather']['date_from']]
    df_tmp = df_tmp[df_tmp['datetimeStr'] < config['weather']['date_till']]

    # add some data
    df_tmp['year'] = df_tmp['date'].dt.year
    df_tmp['month'] = df_tmp['date'].dt.month
    df_tmp['day'] = df_tmp['date'].dt.day
    df_tmp['yyyy_mm'] = df_tmp['date'].dt.strftime('%Y-%m')
    #    dayOfWeek = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    #    df_tmp['weekday'] = df_tmp['date'].dt.dayofweek.map(dayOfWeek)
    df_tmp['rain_season'] = ""

    df_tmp['precip_h'] = df_tmp['precipcover'] / 100 * 24
    df_tmp['precip_mm_h'] = df_tmp['precip'] / df_tmp['precip_h']
    df_tmp['precip_mm_h'] = df_tmp['precip_mm_h'].fillna(0)
    df_tmp['precip_7d'] \
        = df_tmp['precip'].shift(1) \
         + df_tmp['precip'].shift(2) \
         + df_tmp['precip'].shift(3) \
         + df_tmp['precip'].shift(4) \
         + df_tmp['precip'].shift(5) \
         + df_tmp['precip'].shift(6) \
         + df_tmp['precip'].shift(7)
    df_tmp['dry_day'] = df_tmp['precip'].apply(lambda x: 1 if x == 0 else 0).fillna(0)
    df_tmp['wet_day'] = df_tmp['precip'].apply(lambda x: 1 if x > 0 else 0).fillna(0)

    df_tmp['wdir'] = df_tmp['wdir'].fillna(0).round(0).astype(int)

    df_tmp['precip_25'] = df_tmp['precip'] / 25
    df_tmp['precip_25'] = df_tmp['precip_25'].fillna(0).round(0).astype(int) * 25
    df_tmp['maxt_5'] = df_tmp['maxt'] / 5
    df_tmp['maxt_5'] = df_tmp['maxt_5'].fillna(0).round(0).astype(int) * 5
    df_tmp['wdir_10'] = df_tmp['wdir'] / 10
    df_tmp['wdir_10'] = df_tmp['wdir_10'].fillna(0).round(0).astype(int) * 10
    df_tmp['wspd_5'] = df_tmp['wspd'] / 5
    df_tmp['wspd_5'] = df_tmp['wspd_5'].fillna(0).round(0).astype(int) * 5
    df_tmp['humidity_20'] = df_tmp['humidity'] / 20
    df_tmp['humidity_20'] = df_tmp['humidity_20'].fillna(0).round(0).astype(int) * 20

    df_tmp.drop('snow', axis=1, inplace=True)
    df_tmp.drop('snowdepth', axis=1, inplace=True)
    #    df_tmp.drop('windchill', axis=1, inplace = True)
    df_tmp.drop('info', axis=1, inplace=True)
    df_tmp.drop('weathertype', axis=1, inplace=True)
    df_tmp.drop('datetime', axis=1, inplace=True)
    df_tmp.drop('solarenergy', axis=1, inplace=True)
    df_tmp.drop('solarradiation', axis=1, inplace=True)
    df_tmp.drop('address', axis=1, inplace=True)
    df_tmp.drop('location', axis=1, inplace=True)

    if show_data_overview:
        print(df_tmp.info())
        print(df_tmp.head())
        print(df_tmp.tail())
        print(df_tmp.describe())
    return df_tmp


def init_rwh_data(rwh_data):
    """calc rain and usages"""
    rwh_data['collected'] = 0.0
    rwh_data['net_collected_day'] = 0.0
    rwh_data['person_consume'] = 0.0
    rwh_data['garden_consume'] = 0.0
    rwh_data['water_income'] = 0.0
    rwh_data['stored'] = 0.0
    rwh_data['overrun'] = 0.0
    rwh_data['net_overrun'] = 0.0
    rwh_data['tank_overrun'] = 0.0
    return rwh_data


def calc_rwh_collection\
    ( df_tmp
      , roof_name
      , max_filter_throughput
      , max_pipe_throughput
      , effective_collection_area
      , rain_buffer_volume
      , show_data_overview=False):
    """Test the choosen RWH components for heavy rain management
        and return a dataset with all days the system propably couldnt manage the volume of rain
        The max_pipe_throughput is from the gutter to the storm water tank
        and the filter is connected after the storm water tank
        """
    # check throughputs
    if effective_collection_area > 0:
        df_tmp[roof_name+'collected_h'] = effective_collection_area * df_tmp['precip_mm_h'].fillna(0)
        df_tmp[roof_name+'collected_min'] = df_tmp[roof_name+'collected_h'].fillna(0)/60
        df_tmp[roof_name+'collected_day'] = effective_collection_area * df_tmp['precip'].fillna(0)
        df_tmp[roof_name + 'net_collected_day'] = df_tmp[roof_name+'collected_day']

    # check storage capacity
    df_tmp[roof_name+'net_gutter_collected_day'] = df_tmp[roof_name+'collected_day'].fillna(0)
    pipe_rush_yn = df_tmp['datetimeStr'] = df_tmp['datetimeStr']
    if max_pipe_throughput > 0:
        # if the rain is to strong some water cant be collected
        df_tmp[roof_name+'pipe_rush_min'] \
            = df_tmp[roof_name+'collected_min'] - max_pipe_throughput
        df_tmp[roof_name+'pipe_rush_min'] \
            = df_tmp[roof_name+'pipe_rush_min'].apply(lambda x: x if x > 0 else 0).fillna(0)
        df_tmp[roof_name+'pipe_rush_day'] \
            = df_tmp[roof_name+'pipe_rush_min'] * 60 * df_tmp['precip_h'].fillna(0)
        df_tmp[roof_name+'net_gutter_collected_day'] \
            = df_tmp[roof_name+'net_gutter_collected_day']\
                - df_tmp[roof_name+'pipe_rush_min'] * 60 * df_tmp['precip_h'].fillna(0)
        df_tmp[roof_name + 'net_collected_day'] = df_tmp[roof_name+'net_gutter_collected_day']
        df_tmp[roof_name + 'net_overrun'] = df_tmp[roof_name+'pipe_rush_day']

        df_tmp[roof_name+'pipe_rush_day'] = df_tmp[roof_name+'pipe_rush_day'].round(0).astype(int)
        df_tmp[roof_name+'net_gutter_collected_day'] = df_tmp[roof_name+'net_gutter_collected_day'].round(0).astype(int)

        df_tmp.drop(roof_name + 'pipe_rush_min', axis=1, inplace=True)

        print(roof_name+" # of days the gutter overrun = " + str(len(df_tmp[df_tmp[roof_name+'pipe_rush_day'] > 0])))
        pipe_rush_yn = df_tmp[roof_name+'pipe_rush_day'] > 0

    filter_rush_yn = df_tmp['datetimeStr'] = df_tmp['datetimeStr']
    if max_filter_throughput > 0\
    and rain_buffer_volume > 0:
        df_tmp[roof_name+'storm_tank_fill_h'] \
            = rain_buffer_volume / df_tmp[roof_name+'collected_h']
        df_tmp[roof_name+'storm_tank_fill_h'] \
            = df_tmp[roof_name+'storm_tank_fill_h'].apply(lambda x: x if x > 0 else 0).fillna(0)
        df_tmp[roof_name+'collected_plus_h'] \
            = df_tmp[roof_name+'collected_h'] - max_filter_throughput*60
        df_tmp[roof_name+'collected_plus_h'] \
            = df_tmp[roof_name+'collected_plus_h'].apply(lambda x: x if x > 0 else 0).fillna(0)
        df_tmp[roof_name+'filter_rush_day'] \
            = (df_tmp['precip_h'] - df_tmp[roof_name+'storm_tank_fill_h']) * df_tmp[roof_name+'collected_plus_h']
        df_tmp[roof_name+'filter_rush_day'] \
            = df_tmp[roof_name+'filter_rush_day'].apply(lambda x: x if x > 0 else 0).fillna(0)
        df_tmp[roof_name+'net_rain_buffer_collected_day'] \
            = df_tmp[roof_name+'net_gutter_collected_day'] - df_tmp[roof_name+'filter_rush_day']
        df_tmp[roof_name + 'net_collected_day'] = df_tmp[roof_name+'net_rain_buffer_collected_day']
        df_tmp[roof_name + 'net_overrun'] = df_tmp[roof_name+'filter_rush_day']

        df_tmp[roof_name+'filter_rush_day'] = df_tmp[roof_name+'filter_rush_day'].round(0).astype(int)
        df_tmp[roof_name+'net_rain_buffer_collected_day'] = df_tmp[roof_name+'net_rain_buffer_collected_day'].round(0).astype(int)

        df_tmp.drop(roof_name + 'collected_plus_h', axis=1, inplace=True)

        print(roof_name+" # of days the filter overrun = " + str(len(df_tmp[df_tmp[roof_name+'filter_rush_day'] > 0])))
        filter_rush_yn = df_tmp[roof_name+'filter_rush_day'] > 0

    df_tmp['collected'] = df_tmp['collected'] + df_tmp[roof_name + 'collected_day']
    df_tmp['net_collected_day'] = df_tmp['net_collected_day'] + df_tmp[roof_name + 'net_collected_day']
    df_tmp['net_overrun'] = df_tmp['net_overrun'] + df_tmp[roof_name + 'net_overrun']
    df_tmp['collected'] = df_tmp['collected'].round(0).astype(int)
    df_tmp['net_collected_day'] = df_tmp['net_collected_day'].round(0).astype(int)
    df_tmp['net_overrun'] = df_tmp['net_overrun'].round(0).astype(int)

    storm_yn = np.logical_or(pipe_rush_yn, filter_rush_yn)  # filter overrun

    if show_data_overview:
        print(df_tmp.info())
        print(df_tmp.head(60))
        print(df_tmp.tail(60))
        # print(df_tmp.describe())
    return df_tmp, storm_yn


def calc_rwh_system \
    (rwh_data
     , avg_consumer_no
     , consume
     , garden_usage
     , storage_volume
     , tank_reserves
     , show_data_overview=False):
    """calc diverse data for rain water harvesting scenarios
         avg_consumer_no => average number of persons using water (#)
         consume => liter a persons use in average at one day (l/day)
         storage_volume => size of all storage volumes in liter
         garden_usage => liter are used for gardening when there is less rain/humidity (l/day)
    """
    rwh_data['person_consume'] = avg_consumer_no * consume
    stored_last = 0.0
    for index, row in rwh_data.iterrows():
        new_storage = 0.0
        # water just when there is less rain and low humidity
        if row['precip_7d'] < float(config['dwh']['garden_rain_min_mm']) \
                and row['humidity'] < float(config['dwh']['garden_min_humidity']) \
                and stored_last > tank_reserves:
            row['garden_consume'] = garden_usage
            rwh_data.at[index, 'garden_consume'] = garden_usage
        row['water_income'] = row['net_collected_day'] - row['person_consume'] - row['garden_consume']
        rwh_data.at[index, 'water_income'] = row['water_income']
        # calc storage fillstate and overrun
        new_storage = stored_last + row['water_income']
        if 0 < new_storage <= storage_volume:
            rwh_data.at[index, 'stored'] = new_storage
            stored_last = new_storage
        elif new_storage > storage_volume:
            rwh_data.at[index, 'stored'] = storage_volume
            rwh_data.at[index, 'tank_overrun'] = new_storage - storage_volume
            stored_last = storage_volume
        elif new_storage <= 0:
            stored_last = 0
        if 1 <= row['month'] < int(config['dwh']['season_stat_month']):
            rwh_data.at[index, 'rain_season'] = "rs" + str(row['year'] - 1)
        else:
            rwh_data.at[index, 'rain_season'] = "rs" + str(row['year'])

    # round values
    rwh_data.loc[:, ('person_consume')] = rwh_data['person_consume'].round(0).astype(int)
    rwh_data.loc[:, ('garden_consume')] = rwh_data['garden_consume'].round(0).astype(int)
    rwh_data.loc[:, ('water_income')] = rwh_data['water_income'].round(0).astype(int)
    rwh_data.loc[:, ('stored')] = rwh_data['stored'].round(0).astype(int)
    rwh_data.loc[:, ('tank_overrun')] = rwh_data['tank_overrun'].round(0).astype(int)

    rwh_data['overrun'] = rwh_data['net_overrun'] + rwh_data['tank_overrun']
    tank_overrun_yn = rwh_data['tank_overrun'] > 0
    print(" # of days the tank overrun = " + str(len(rwh_data[rwh_data['tank_overrun'] > 0])))

    rwh_data['store_filled_pct'] = rwh_data['stored'] / storage_volume
    rwh_data['store_filled_grp'] = rwh_data['store_filled_pct']. \
        apply(lambda x: '00' if x == 0.0 \
        else ('00' if x == 0.0
              else ('01-10' if 0.0 < x <= 0.1
                    else ('11-33' if 0.1 < x <= 0.33
                          else ('34-66' if 0.33 < x <= 0.66
                                else '67+'
                                )
                          )
                    )
              )
              )

    if show_data_overview:
        print(rwh_data.info())
        print(rwh_data.head(60))
        print(rwh_data.tail(60))
        # print(rwh_data.describe())
    return rwh_data, tank_overrun_yn


def group_rwh_data_ym(df, group_fields, show_data_overview=False):
    """calc diverse data gruped by year month"""

    def quantile(x, n):
        return x.quantile(n)

    df_tmp = df.groupby(group_fields, as_index=True, sort=False) \
        .agg(yyyy_mm=("yyyy_mm", "min")
             , days=("yyyy_mm", "count")
             , precip_sum=("precip", "sum")
             , precip_min=("precip", "min")
             , precip_q1=("precip", lambda x: np.percentile(x, q=25))
             , precip_avg=("precip", np.mean)
             , precip_med=("precip", np.median)
             , precip_q3=("precip", lambda x: np.percentile(x, q=75))
             , precip_max=("precip", "max")
             , precip_std=("precip", np.std)
             , dry_day_sum=("dry_day", "sum")
             , wet_day_sum=("wet_day", "sum")
             , collected_sum=("collected", "sum")
             , net_collected_sum=("net_collected_day", "sum")
             , precip_h_sum=("precip_h", "sum")
             , precip_h_q05=("precip_h", lambda x: np.percentile(x, q=5))
             , precip_h_avg=("precip_h", np.mean)
             , precip_h_med=("precip_h", np.median)
             , precip_h_q95=("precip_h", lambda x: np.percentile(x, q=95))
             , precip_mm_h_min=("precip_mm_h", "min")
             , precip_mm_h_avg=("precip_mm_h", np.mean)
             , precip_mm_h_max=("precip_mm_h", "max")
             , person_sum=("person_consume", "sum")
             , garden_sum=("garden_consume", "sum")
             , stored_min=("stored", "min")
             , stored_max=("stored", "max")
             , stored_grp_min=("store_filled_grp", "min")
             , stored_grp_max=("store_filled_grp", "max")
             , overrun_sum=("overrun", "sum")
             , net_overrun_sum=("net_overrun", "sum")
             , tank_overrun_sum=("tank_overrun", "sum")
             , temp_min=("temp", "min")
             , temp_avg=("temp", np.mean)
             , temp_max=("maxt", "max")
             , precipcover_min=("precipcover", "min")
             , precipcover_avg=("precipcover", np.mean)
             , precipcover_max=("precipcover", "max")
             , cloudcover_min=("cloudcover", "min")
             , cloudcover_avg=("cloudcover", np.mean)
             , cloudcover_max=("cloudcover", "max")
             , humidity_min=("humidity", "min")
             , humidity_avg=("humidity", np.mean)
             , humidity_max=("humidity", "max")
             , wspd_min=("wspd", "min")
             , wspd_avg=("wspd", np.mean)
             , wspd_max=("wspd", "max")
             , wgust_min=("wgust", "min")
             , wgust_avg=("wgust", np.mean)
             , wgust_max=("wgust", "max")
             )
    df_tmp['precip_avg'] = df_tmp['precip_avg'].round(2)
    df_tmp['precip_std'] = df_tmp['precip_std'].round(2)
    df_tmp['precip_h_sum'] = df_tmp['precip_h_sum'].round(0)
    df_tmp['precip_h_q05'] = df_tmp['precip_h_q05'].round(1)
    df_tmp['precip_h_avg'] = df_tmp['precip_h_avg'].round(1)
    df_tmp['precip_h_med'] = df_tmp['precip_h_med'].round(1)
    df_tmp['precip_h_q95'] = df_tmp['precip_h_q95'].round(1)
    df_tmp['precip_mm_h_min'] = df_tmp['precip_mm_h_min'].round(1)
    df_tmp['precip_mm_h_avg'] = df_tmp['precip_mm_h_avg'].round(1)
    df_tmp['precip_mm_h_max'] = df_tmp['precip_mm_h_max'].round(1)
    df_tmp['temp_avg'] = df_tmp['temp_avg'].round(1)
    df_tmp['precipcover_avg'] = df_tmp['precipcover_avg'].round(1)
    df_tmp['cloudcover'] = df_tmp['cloudcover_avg'].round(1)
    df_tmp['humidity_avg'] = df_tmp['humidity_avg'].round(1)
    df_tmp['wspd_avg'] = df_tmp['wspd_avg'].round(1)
    df_tmp['wgust_avg'] = df_tmp['wgust_avg'].round(1)

    df_desc = pd.DataFrame({'precip': df['precip'].describe()
                               , 'precip_h': df['precip_h'].describe()
                               , 'precip_mm_h': df['precip_mm_h'].describe()
                               , 'humidity': df['humidity'].describe()
                               , 'precipcover': df['precipcover'].describe()
                               , 'cloudcover': df['cloudcover'].describe()
                               , 'temp': df['temp'].describe()
                               , 'maxt': df['maxt'].describe()
                               , 'wspd': df['wspd'].describe()
                               , 'wgust': df['wgust'].describe()
                               , 'windchill': df['windchill'].describe()})
    df_desc.loc['05%'] = df.quantile(0.05).round(1)
    df_desc.loc['95%'] = df.quantile(0.95).round(1)
    df_desc.loc['97,5%'] = df.quantile(0.975).round(1)
    df_desc.loc['99%'] = df.quantile(0.99).round(1)
    df_desc.loc['dtype'] = df_desc.dtypes
    df_desc.loc['% count'] = df.isnull().mean().round(4)
    df_desc = df_desc.reset_index()
    if show_data_overview:
        print(df_tmp.info())
        print(df_tmp.head(60))
        print(df_tmp.tail(60))
        # print(df_tmp.describe())
    return df_tmp, df_desc


def main():
    df = load_csv(False)
    df = init_rwh_data(df)

    # default RWH parameters
    df, df_storm_gr\
        = calc_rwh_collection\
            (df, "gr_"
            , 35 # l/minute max_filter_throughput
            , 870 # l/minute max_pipe_throughput
            , 93 * 0.85 # effective_collection_area
            , 3000 # rain_buffer_volume
            , False
            )
    df, df_storm_mr\
        = calc_rwh_collection\
            (df, "mr_"
            , 999 # l/minute max_filter_throughput
            , 870 # l/minute max_pipe_throughput
            , 189 * 0.5  # effective_collection_area
            , 3000 # rain_buffer_volume
            , False
            )
    df, df_tank\
        = calc_rwh_system(df  # rwh_data
                         , eval(config['dwh']['avg_consumer_no'])
                         , eval(config['dwh']['person_consume'])
                         , eval(config['dwh']['garden_usage'])
                         , eval(config['dwh']['storage_volume'])
                         , eval(config['dwh']['tank_reserves'])
                         , False  # show_data_overview
                         )
    print(df[np.logical_or(np.logical_or(df_storm_gr, df_storm_mr), df_tank)].head(30))
    del df_storm_gr
    del df_storm_mr
    del df_tank


    df_ym, df_total = group_rwh_data_ym(df, ['year', 'month'], False)
    print("Summary of Totals")
    print(df_total)

    return df, df_total, df_ym


"""Main run section"""
main()
