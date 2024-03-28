"""collect weather data from visualcrossing API
author: Matthis (Dev4Data-github@online.ms)
description:
    The weather data is collected and saved in the weather_file
    Parameters are taken from the *.ini files (dnt forget to setup the key.ini with the visualcrossing API key)

functions:
    get_latlon_weather_json - method to access the visualcrossing API
    get_new_weather_data - method to do one API call with the desired data and limited API requirements
    get_weather_data_run - main method to call the get_new_weather_data several times to use daily limits
return:
     save the collected data in the  weather_file
"""
import pandas as pd
import numpy as np
import datetime
import requests
import csv

from src import setup_environment as env


# constants
"""some settings are saved in local variables here
    you have to change the keys array here according to the key.ini,
    when you a dont use the same number of keys as shown 
"""
config = env.get_config()
PROJECT_ROOT = env.get_project_root()
weather_file = "{}/{}".format(PROJECT_ROOT, config['files']['weatherFile'])
remain_rows = int(config['visualcrossing']['max_rows_per_day'])

keys = []
for i in range(1, 10):
    try:
        keys.append('&key=' + config['visualcrossing_private']['key'+str(i)])
    except:
        break
print("found {} keys in keys.ini file".format(len(keys)))


def get_latlon_weather_json(
    latlon_in,
    date_min_in,
    date_max_in,
    use_key_num=0
):
    """method to get historical weather data for one lat+lon position between start and end-date
            as daily aggregate in metric unit.
            Data is return as aggregate from the closest stations. If the point has no direct station.

        parameters:
            latlon_in position array with latitude and longitude (like [12.34567,12.34567])
            date_min_in: data of the first entry
        return:
            json object with data and column description
    """
    API_key = keys[use_key_num]
    hist_weather_url = config['visualcrossing']['hist_weather_url']
    query_location = '&location={},{}'.format(latlon_in[0], latlon_in[1])
    query_date = '&startDateTime=' + date_min_in + 'T00:00:00&endDateTime=' + date_max_in + 'T00:00:00'
    query_type_params = '&aggregateHours=24&unitGroup=metric&outputDateTimeFormat=yyyy-MM-dd'\
                        + '&dayStartTime=0:0:00&dayEndTime=0:0:00' + query_date\
                        + '&unitGroup=metric&locationMode=single&contentType=json&shortColumnNames=true'
    url = hist_weather_url + query_type_params + query_location + API_key
    print(' - Running query URL: ', url)
    # os.system("pause")
    try:
        response = requests.get(url)
        # response = data_scraper('GET', url)
    except requests.exceptions.RequestException as e:
        print('RequestException: ', e)
        return
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print('HTTPError: ', e)
        return

    json_obj = response.json()
    return json_obj


def get_new_weather_data(
    num_rows=99,
    use_key_num=0,
    collect_data_before_now="y"
):
    """method that find the from-date and to-date to limit the weather data API call,
        it just load data where the weather data is not collected yet
        it calls the get_latlon_weather_json to ask for new weather data

        parameter:
            num_rows - get this number of rows from API (from-to dates are choosen according it)
            use_key_num - which key should be used (depending on the specified keys in the ini file)
            collect_data_before_now
                        - y=>collect data older than all already existing data
                        - other-parameters=>collect data newer than all already existing data
        return:
                 remaining_cost - integer/# of remaining API calls
                 filled_lines - integer/# of data that was collected
    """
    try:
        df = pd.read_csv(weather_file)
    except FileNotFoundError:
        print("File not found. "+weather_file)
        return
    except pd.errors.EmptyDataError:
        print("No data in "+weather_file)
        return
    except pd.errors.ParserError:
        print("Parse error for file "+weather_file)
        return

    df['date'] = df['date'].astype({'date': 'datetime64[ns]'})
    df.reset_index(inplace=True)
    df_tmp = df.groupby(['location'], as_index=False, sort=False) \
                        .agg({'date': ['min', 'max']})
    df_tmp.columns = ["_".join(x) for x in df_tmp.columns.ravel()]
    print(df_tmp)
    filled_lines = 0
    remaining_cost = remain_rows
    for loc in df_tmp.index:
        # get smallest and highest date for location (max x rows)
        location = df_tmp["location_"][loc]
        if collect_data_before_now == "y":
            df_date_max_tmp = df_tmp["date_min"][loc] - datetime.timedelta(1)
            df_date_max = df_date_max_tmp.strftime("%Y-%m-%d")
            df_date_min_tmp = df_tmp["date_min"][loc]\
                - datetime.timedelta(num_rows)
            df_date_min = df_date_min_tmp.strftime("%Y-%m-%d")
        else:
            df_date_min_tmp = df_tmp["date_max"][loc] + datetime.timedelta(1)
            df_date_min = df_date_min_tmp.strftime("%Y-%m-%d")
            df_date_max_tmp = \
                np.minimum((datetime.datetime.today() - datetime.timedelta(1))
                             , df_tmp["date_max"][loc]
                                + datetime.timedelta(int(num_rows)))
            df_date_max = df_date_max_tmp.strftime("%Y-%m-%d")

        # latitude+longitude of location
        latlng = [config['location']['latitude'], config['location']['longitude']]
        # get the weather data as csv
        print("geo={}; date min={} and max={}".format(latlng, df_date_min, df_date_max))
        json_data = get_latlon_weather_json(latlng, df_date_min, df_date_max, use_key_num)
#        print(json_data['location'])

        if "messages" in json_data:
            msg = json_data["messages"]
        elif "message" in json_data:
            msg = json_data["message"]
        else:
            msg = "nan"
        info = json_data['location']['values'][0]['info']
        if info == "No data available":
            print("msg: {}".format(info))
            return json_data['remainingCost'], 0

        if msg == None\
        or msg == "nan":
            if "remainingCost" in json_data:
                remaining_cost = int(json_data['remainingCost'])
                print("remaining_cost: {}, at key: {}".format(remaining_cost, use_key_num))
                if remaining_cost >= 0:
                    if "address" in json_data:
                        locations = json_data['address']
                    elif "addresses" in json_data:
                        locations = json_data['addresses']
                    else:
                        locations = json_data['location']

                    df_json = pd.json_normalize(locations, ['values'], meta=['id'])
                    df_json.rename(columns={"id": "address"}, inplace=True)
                    df_json['date'] = df_json['datetimeStr']
                    df_json['date'] = df_json['date'].astype({'date': 'datetime64[ns]'})
                    df_json['location'] = location
                    df_json.to_csv(weather_file, sep=',', encoding='utf-8', mode='a+', header=False, index=False,
                                   quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
                    #print(df_json.info())
                    #print(df_json.head())
                    df_snapshot = df_json[['location','date','address','wdir','maxt','conditions']].head(1)
                    # df_snapshot.append(df[['location','date','address','wdir','maxt','conditions']].tail(1))
                    pd.concat([df_snapshot, df[['location','date','address','wdir','maxt','conditions']]]).tail(1)
                    print("weatherdata # added {}".format(df.shape[0]))
                #    print(df_snapshot)
                    filled_lines = df_json.shape[0]
        else:
            print("msg: {}".format(msg))
            return json_data['remainingCost'], 0
    return remaining_cost, filled_lines


def get_weather_data_run():
    """method as a skull to loop over the API keys defined
            and  loop also x times to collect data according to the daily limit and the limit per API call

        parameter:
        return:
                 just display some summary data
    """
    # loop both API_keys
    print("start gather weatherdata")
    for key in range(len(keys)):
        remain_rows_tmp = remain_rows
        sum_filled = 0
        # loop x times each time with 50-100 rows to full fill the limits of API calls
        for i in range(int(config['visualcrossing']['loops'])):
            get_rows = np.minimum(int(config['visualcrossing']['rows_per_run'])
                                  , remain_rows_tmp)
            remain_rows_tmp, filled \
                = get_new_weather_data( num_rows=get_rows
                                       , use_key_num=key
                                       , collect_data_before_now="n")
            sum_filled += filled
            if filled == 0:
                print("{} new # of weatherdata with key={} remain_rows={} "\
                      .format(sum_filled, key, remain_rows_tmp))
                break


"""Main section"""
if __name__ == '__main__':
    get_weather_data_run()
