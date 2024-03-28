"""wrangle the data and calculate the rain water harvesting (RWH) system
author: Matthis (Dev4Data-github@online.ms)
description:
    clean, wrangle and enrich the weather data
    calculate the RWH systems with the parameters from the settings.ini
    and aggregate it for later analyses and visualisation

functions:
    load_csv - load the weatherdata from csv file and calculate required and some need additional data
    init_rwh_data - initialise empty fields for later filling with rwh data
    calc_rwh_collection - calculate the collected amount of water from one roof
    calc_rwh_system - calculate the performance of the RWH design according to the specified parameters
    group_rwh_data_ym - group the data by year-month and calculate some aggregates
    main - method process to order the calls of the methods and set parameters to create a RWH system
return:
    three DataFrames
    df - dataset with all calculated data for the RWH system
    df_total - summary (like df.describe()) to show a description of the DataFrame
    df_ym - aggregated dataset on year-month basis

"""
import pandas as pd
import numpy as np

from src import setup_environment as env


"""Show parameters"""
env.set_pd_environments()
# constants
config = env.get_config()


def load_csv() \
        -> pd.DataFrame:
    """method that load data from weather csv
        parameters:
        return:
            DataFrame with the data
    """
    PROJECT_ROOT = env.get_project_root()
    file_path = "{}/{}".format(PROJECT_ROOT, config['files']['weatherFile'])
    df_tmp: pd.DataFrame = pd.read_csv(file_path)

    return df_tmp


def transform_data(df_tmp: pd.DataFrame, show_data_overview: bool = False) \
            -> pd.DataFrame:
    """method that wrangle and enrich data from weather csv
            datetime data is split in different fields and wrangled
            precipitation data is enriched
            and some data is also categorized in ranges
            unnecessary fields are also dropped

        parameters:
            show_data_overview - True => some prints to show an overview of the data
        return:
            DataFrame with the data
    """
    """wrangle and clean data"""
    df_tmp['date'] = df_tmp['date'].astype({'date': 'datetime64[ns]'})
    # df_tmp = df_tmp.set_index(['date'])
    df_tmp = df_tmp.sort_values('date')

    # limit the time frame
    df_tmp = df_tmp[df_tmp['datetimeStr'] >= config['weather']['date_from']]
    df_tmp = df_tmp[df_tmp['datetimeStr'] < config['weather']['date_till']]

    # add some data
    df_tmp['year'] = df_tmp['date'].dt.year.astype(np.int64)
    df_tmp['month'] = df_tmp['date'].dt.month.astype(np.int64)
    df_tmp['day'] = df_tmp['date'].dt.day.astype(np.int64)
    df_tmp['week'] = df_tmp['date'].dt.isocalendar().week.astype(np.int64)
    df_tmp['yyyy_mm'] = df_tmp['date'].dt.strftime('%Y-%m')
    #    dayOfWeek = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    #    df_tmp['weekday'] = df_tmp['date'].dt.dayofweek.map(dayOfWeek)
    df_tmp['rain_season'] = ""

    df_tmp['precip'] = df_tmp['precip'].astype(float)
    df_tmp['precip_h'] = (df_tmp['precipcover'] / 100.0 * 24.0).astype(float)
    df_tmp['precip_h'] = df_tmp['precip_h'].astype(float)
    df_tmp['precip_mm_h'] = (df_tmp['precip'] / df_tmp['precip_h']).astype(float)
    df_tmp['precip_mm_h'] = df_tmp['precip_mm_h'].fillna(0)
    df_tmp['precip_7d'] \
        = df_tmp['precip'].shift(1) \
        + df_tmp['precip'].shift(2) \
        + df_tmp['precip'].shift(3) \
        + df_tmp['precip'].shift(4) \
        + df_tmp['precip'].shift(5) \
        + df_tmp['precip'].shift(6) \
        + df_tmp['precip'].shift(7)
    df_tmp['dry_day'] = df_tmp['precip'].apply(lambda x: 1 if x == 0 else 0).fillna(0).astype(np.int64)
    df_tmp['wet_day'] = df_tmp['precip'].apply(lambda x: 1 if x > 0 else 0).fillna(0).astype(np.int64)
    df_tmp['precip_grp'] = df_tmp['precip']. \
        apply(lambda x: '00' if x == 0.0 \
            else ('00-02' if 0.0 < x <= 2.0
                  else ('03-05' if 0.2 < x <= 5.0
                        else ('06-10' if 5.0 < x <= 10.0
                              else ('11-20' if 10.0 < x <= 20.00
                                    else '21+'
                                    )
                              )
                        )
                  )
              )

    df_tmp['wdir'] = df_tmp['wdir'].fillna(0).round(0).astype(np.int64)

    df_tmp['precip_25'] = (df_tmp['precip'] / 25).astype(float)
    df_tmp['precip_25'] = df_tmp['precip_25'].fillna(0.0).round(0).astype(np.int64) * 25
    df_tmp['maxt_5'] = (df_tmp['maxt'] / 5).astype(float)
    df_tmp['maxt_5'] = df_tmp['maxt_5'].fillna(0.0).round(0).astype(np.int64) * 5
    df_tmp['wdir_10'] = (df_tmp['wdir'] / 10).astype(float)
    df_tmp['wdir_10'] = df_tmp['wdir_10'].fillna(0.0).round(0).astype(np.int64) * 10
    df_tmp['wspd_5'] = (df_tmp['wspd'] / 5).astype(float)
    df_tmp['wspd_5'] = df_tmp['wspd_5'].fillna(0.0).round(0).astype(np.int64) * 5
    df_tmp['humidity_20'] = (df_tmp['humidity'] / 20).astype(float)
    df_tmp['humidity_20'] = df_tmp['humidity_20'].fillna(0.0).round(0).astype(np.int64) * 20

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


def init_rwh_data(rwh_data: pd.DataFrame) -> pd.DataFrame:
    """initialise empty fileds for RWH calculations"""
    rwh_data['collected'] = float(0.0)
    rwh_data['net_collected_day'] = float(0.0)
    rwh_data['person_consume'] = float(0.0)
    rwh_data['garden_consume'] = float(0.0)
    rwh_data['water_income'] = float(0.0)
    rwh_data['stored'] = float(0.0)
    rwh_data['overrun'] = float(0.0)
    rwh_data['net_overrun'] = float(0.0)
    rwh_data['tank_overrun'] = float(0.0)
    return rwh_data


def calc_rwh_collection(
    df_tmp: pd.DataFrame
    , roof_name: str
    , max_pipe_throughput: float
    , max_filter_throughput: float
    , effective_collection_area: float
    , rain_buffer_volume: float
    , show_data_overview: bool = False
    )\
        -> [pd.DataFrame, pd.DataFrame]:
    """method to calculate the performance of a RWH component
            calculates the collected water and the water that overruns from gutter or filter
            The overrun from the storage is calculated in the calc_rwh_system method.

        parameters:
            df_tmp - DataFrame with the weather data enriched with data from load_csv method
            roof_name - pre-column name for the roof (mainly for differentiation in the dataset)
            max_pipe_throughput - what is the smallest pipe throughput (liter/minute)
                                    provided by your gutter and downspout
            max_filter_throughput - what is the smallest filter throughput (liter/minute)
            effective_collection_area - the effective collection area of your roof
            rain_buffer_volume - size of the rain buffer tank between pipes and filter
            show_data_overview - True => some prints to show an overview of the data
        return:
            a enriched dataset with the collected water and also show when the system couldnt manage the volume of rain
    """
    # check throughputs
    if effective_collection_area > 0.0:
        df_tmp[roof_name+'collected_h'] = effective_collection_area * df_tmp['precip_mm_h'].fillna(0.0)
        df_tmp[roof_name+'collected_min'] = df_tmp[roof_name+'collected_h'].fillna(0.0)/60
        df_tmp[roof_name+'collected_day'] = effective_collection_area * df_tmp['precip'].fillna(0.0)
        df_tmp[roof_name + 'net_collected_day'] = df_tmp[roof_name+'collected_day']

    # check storage capacity
    df_tmp[roof_name+'net_gutter_collected_day'] = df_tmp[roof_name+'collected_day'].fillna(0.0)
    pipe_rush_yn = df_tmp['datetimeStr'] = df_tmp['datetimeStr']
    if max_pipe_throughput > 0.0:
        # if the rain is to strong some water cant be collected
        df_tmp[roof_name+'pipe_rush_min'] \
            = df_tmp[roof_name+'collected_min'] - max_pipe_throughput
        df_tmp[roof_name+'pipe_rush_min'] \
            = df_tmp[roof_name+'pipe_rush_min'].apply(lambda x: x if x > 0.0 else 0.0).fillna(0.0)
        df_tmp[roof_name+'pipe_rush_day'] \
            = df_tmp[roof_name+'pipe_rush_min'] * 60 * df_tmp['precip_h'].fillna(0.0)
        df_tmp[roof_name+'net_gutter_collected_day'] \
            = df_tmp[roof_name+'net_gutter_collected_day']\
                - df_tmp[roof_name+'pipe_rush_min'] * 60 * df_tmp['precip_h'].fillna(0.0)
        df_tmp[roof_name + 'net_collected_day'] = df_tmp[roof_name+'net_gutter_collected_day']
        df_tmp[roof_name + 'net_overrun'] = df_tmp[roof_name+'pipe_rush_day']

        df_tmp[roof_name+'pipe_rush_day'] = df_tmp[roof_name+'pipe_rush_day'].round(0).astype(np.int64)
        df_tmp[roof_name+'net_gutter_collected_day'] \
            = df_tmp[roof_name+'net_gutter_collected_day'].round(0).astype(np.int64)

        df_tmp.drop(roof_name + 'pipe_rush_min', axis=1, inplace=True)

        print(roof_name+" # of days the gutter overrun = " + str(len(df_tmp[df_tmp[roof_name+'pipe_rush_day'] > 0])))
        pipe_rush_yn = df_tmp[roof_name+'pipe_rush_day'] > 0.0

    filter_rush_yn = df_tmp['datetimeStr'] = df_tmp['datetimeStr']
    if max_filter_throughput > 0.0\
    and rain_buffer_volume > 0.0:
        df_tmp[roof_name+'storm_tank_fill_h'] \
            = rain_buffer_volume / df_tmp[roof_name+'collected_h']
        df_tmp[roof_name+'storm_tank_fill_h'] \
            = df_tmp[roof_name+'storm_tank_fill_h'].apply(lambda x: x if x > 0.0 else 0.0).fillna(0.0)
        df_tmp[roof_name+'collected_plus_h'] \
            = df_tmp[roof_name+'collected_h'] - max_filter_throughput*60
        df_tmp[roof_name+'collected_plus_h'] \
            = df_tmp[roof_name+'collected_plus_h'].apply(lambda x: x if x > 0.0 else 0.0).fillna(0.0)
        df_tmp[roof_name+'filter_rush_day'] \
            = (df_tmp['precip_h'] - df_tmp[roof_name+'storm_tank_fill_h']) * df_tmp[roof_name+'collected_plus_h']
        df_tmp[roof_name+'filter_rush_day'] \
            = df_tmp[roof_name+'filter_rush_day'].apply(lambda x: x if x > 0.0 else 0.0).fillna(0.0)
        df_tmp[roof_name+'net_rain_buffer_collected_day'] \
            = df_tmp[roof_name+'net_gutter_collected_day'] - df_tmp[roof_name+'filter_rush_day']
        df_tmp[roof_name + 'net_collected_day'] = df_tmp[roof_name+'net_rain_buffer_collected_day']
        df_tmp[roof_name + 'net_overrun'] = df_tmp[roof_name+'filter_rush_day']

        df_tmp[roof_name+'filter_rush_day'] = df_tmp[roof_name+'filter_rush_day'].round(0).astype(np.int64)
        df_tmp[roof_name+'net_rain_buffer_collected_day'] = df_tmp[roof_name+'net_rain_buffer_collected_day'].round(0).astype(np.int64)

        df_tmp.drop(roof_name + 'collected_plus_h', axis=1, inplace=True)

        print(roof_name+" # of days the filter overrun = " + str(len(df_tmp[df_tmp[roof_name+'filter_rush_day'] > 0])))
        filter_rush_yn = df_tmp[roof_name+'filter_rush_day'] > 0

    df_tmp['collected'] = df_tmp['collected'] + df_tmp[roof_name + 'collected_day']
    df_tmp['net_collected_day'] = df_tmp['net_collected_day'] + df_tmp[roof_name + 'net_collected_day']
    df_tmp['net_overrun'] = df_tmp['net_overrun'] + df_tmp[roof_name + 'net_overrun']
    df_tmp['collected'] = df_tmp['collected'].round(0).astype(np.int64)
    df_tmp['net_collected_day'] = df_tmp['net_collected_day'].round(0).astype(np.int64)
    df_tmp['net_overrun'] = df_tmp['net_overrun'].round(0).astype(np.int64)

    storm_yn = np.logical_or(pipe_rush_yn, filter_rush_yn)  # filter overrun

    if show_data_overview:
        print(df_tmp.info())
        print(df_tmp.head(60))
        print(df_tmp.tail(60))
        # print(df_tmp.describe())
    return df_tmp, storm_yn


def calc_rwh_system(
    rwh_data: pd.DataFrame
    , avg_consumer_no: float
    , consume: float
    , garden_usage: float
    , storage_volume: float
    , tank_reserves: float
    , show_data_overview: bool = False
    )\
        -> [pd.DataFrame, pd.DataFrame]:
    """method to calculate scenario data for rain water harvesting components
            It calculates the fill state of the storage at a particular date
            And how much water overrun because the storages are full

        parameters:
            rwh_data - DataFrame with weather data and RWH component data
            avg_consumer_no => average number of persons using water (#)
            consume => liter a persons use in average at one day (l/day)
            garden_usage => liter are used for gardening when there is less rain/humidity (l/day)
            storage_volume => size of all storage volumes in liter
            tank_reserves => how empty can be the tank before you stop watering the garden
            show_data_overview - True => some prints to show an overview of the data
       return:
            a enriched dataset with the RWH system performance
    """
    rwh_data['person_consume'] = avg_consumer_no * consume
    stored_last = 0.0
    for index, row in rwh_data.iterrows():
        new_storage = 0.0
        # water just when there is less rain and low humidity
        if row['precip_7d'] < float(config['rwh']['garden_rain_min_mm']) \
                and row['humidity'] < float(config['rwh']['garden_min_humidity']) \
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
        if 1 <= row['month'] < int(config['rwh']['season_stat_month']):
            rwh_data.at[index, 'rain_season'] = "rs" + str(row['year'] - 1)
        else:
            rwh_data.at[index, 'rain_season'] = "rs" + str(row['year'])

    # round values
    rwh_data['person_consume'] = rwh_data['person_consume'].apply(lambda x: int(round(x, 0)))
    rwh_data['garden_consume'] = rwh_data['garden_consume'].apply(lambda x: int(round(x, 0)))
    rwh_data['water_income'] = rwh_data['water_income'].apply(lambda x: int(round(x, 0)))
    rwh_data['stored'] = rwh_data['stored'].apply(lambda x: int(round(x, 0)))
    rwh_data['tank_overrun'] = rwh_data['tank_overrun'].apply(lambda x: int(round(x, 0)))

    rwh_data['overrun'] = rwh_data['net_overrun'] + rwh_data['tank_overrun']
    tank_overrun_yn = rwh_data['tank_overrun'] > 0
    print(" # of days the tank overrun = " + str(len(rwh_data[rwh_data['tank_overrun'] > 0])))

    rwh_data['store_filled_pct'] = (rwh_data['stored'] / storage_volume).astype(float)
    rwh_data['store_filled_grp'] = rwh_data['store_filled_pct']. \
        apply(lambda x: '00' if x == 0.0 \
            else ('01-10' if 0.0 < x <= 0.1
                    else ('11-33' if 0.1 < x <= 0.33
                          else ('34-66' if 0.33 < x <= 0.66
                                else '67+'
                                )
                          )
                    )
               )
    rwh_data['tank_empty'] = rwh_data['store_filled_pct'].apply(lambda x: 1 if x == 0.0 else 0).fillna(0)
    rwh_data['tank_low'] = rwh_data['store_filled_pct'].apply(lambda x: 1 if x <= 0.1 else 0).fillna(0)

    if show_data_overview:
        print(rwh_data.info())
        print(rwh_data.head(60))
        print(rwh_data.tail(60))
        # print(rwh_data.describe())
    return rwh_data, tank_overrun_yn


def date_group_rwh_data(df_in: pd.DataFrame,
                        group_fields: list,
                        show_data_overview: bool = False) \
        -> [pd.DataFrame, pd.DataFrame]:
    """method to group the RWH dataset by defined fields
            It calculates some aggregates and other statistical data

        parameters:
            df - DataFrame with weather data and RWH system data
            group_fields => fields that should be used for grouping
            show_data_overview - True => some prints to show an overview of the data
        return:
            a grouped dataset with statistical data
    """
    df_tmp = df_in.groupby(group_fields, as_index=True, sort=False) \
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
             , tank_empty_sum=("tank_empty", "sum")
             , tank_low_sum=("tank_low", "sum")
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

    df_desc = pd.DataFrame({'precip': df_in['precip'].describe().astype(float)
                               , 'precip_h': df_in['precip_h'].describe().astype(float)
                               , 'precip_mm_h': df_in['precip_mm_h'].describe().astype(float)
                               , 'humidity': df_in['humidity'].describe().astype(float)
                               , 'precipcover': df_in['precipcover'].describe().astype(float)
                               , 'cloudcover': df_in['cloudcover'].describe().astype(float)
                               , 'temp': df_in['temp'].describe().astype(float)
                               , 'maxt': df_in['maxt'].describe().astype(float)
                               , 'wspd': df_in['wspd'].describe().astype(float)
                               , 'wgust': df_in['wgust'].describe().astype(float)
                               , 'windchill': df_in['windchill'].describe().astype(float)})
    df_desc = df_desc.reset_index()
    # TODO: RuntimeWarning in subtract
    df_desc.loc['05%'] = (df_in.quantile(0.05, numeric_only=True).round(1)).astype(float)
    df_desc.loc['95%'] = (df_in.quantile(0.95, numeric_only=True).round(1)).astype(float)
    df_desc.loc['97,5%'] = (df_in.quantile(0.975, numeric_only=True).round(1)).astype(float)
    df_desc.loc['99%'] = (df_in.quantile(0.99, numeric_only=True).round(1)).astype(float)
    df_desc.loc['dtype'] = df_desc.dtypes
    df_desc.loc['% count'] = df_in.isnull().mean().round(4).astype(float)
    if show_data_overview:
        print(df_tmp.info())
        print(df_tmp.head(60))
        print(df_tmp.tail(60))
        # print(df_tmp.describe())
    return df_tmp, df_desc


def main() \
        -> [pd.DataFrame, pd.DataFrame,
            pd.DataFrame, pd.DataFrame,
            pd.DataFrame]:
    """main method to order the method calls and fill the parameters from the config file"""
    df: pd.DataFrame = load_csv()
    df = transform_data(df, False)
    df = init_rwh_data(df)

    # default RWH parameters
    df, df_storm_gr\
        = calc_rwh_collection\
            (df, config['rwh']['r1_name']
            , float(eval(config['rwh']['r1_max_pipe_throughput']))
            , float(eval(config['rwh']['r1_max_filter_throughput']))
            , float(eval(config['rwh']['r1_effective_collection_area']))
            , float(eval(config['rwh']['r1_rain_buffer_volume']))
            , False
            )
    df, df_storm_mr\
        = calc_rwh_collection\
            (df, config['rwh']['r2_name']
            , float(eval(config['rwh']['r2_max_pipe_throughput']))
            , float(eval(config['rwh']['r2_max_filter_throughput']))
            , float(eval(config['rwh']['r2_effective_collection_area']))
            , float(eval(config['rwh']['r2_rain_buffer_volume']))
            , False
            )
    df, df_tank\
        = calc_rwh_system(df  # rwh_data
                         , float(eval(config['rwh']['avg_consumer_no']))
                         , float(eval(config['rwh']['person_consume']))
                         , float(eval(config['rwh']['garden_usage']))
                         , float(eval(config['rwh']['storage_volume']))
                         , float(eval(config['rwh']['tank_reserves']))
                         , False  # show_data_overview
                         )
    print("Example RWH data")
    print(df[np.logical_or(np.logical_or(df_storm_gr, df_storm_mr), df_tank)].tail(10).transpose())
    print()
    del df_storm_gr
    del df_storm_mr
    del df_tank

    df_y, df_total_y = date_group_rwh_data(df, ['year'], False)
    df_ym, df_total = date_group_rwh_data(df, ['year', 'month'], False)
    df_yw, df_total_w = date_group_rwh_data(df, ['year', 'week'], False)
    print("Description of all numeric values")
    print(df_total)
    print()
    print("Summary of Figures per Yearly")
    print(df_y.tail(10).transpose())
    # print(df_y.info())

    return df, df_total, df_ym, df_y, df_yw


"""Main run section"""
if __name__ == '__main__':
    main()
