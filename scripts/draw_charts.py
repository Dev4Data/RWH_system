"""show and save diagrams to understand the weather data and performance of the RWH components
author: Matthis (Dev4Data-github@online.ms)
description:
    diverse methods to show diagrams with weather data
    and RWH performance data

functions:
    load_csv - load the weatherdata from csv file and calculate required and some need additional data
    init_rwh_data - initialise empty fields for later filling with rwh data
    calc_rwh_collection - calculate the collected amount of water from one roof
    calc_rwh_system - calculate the performance of the RWH design according to the specified parameters
    group_rwh_data_ym - group the data by year-month and calculate some aggregates
    main - method process to order the calls of the methods and set parameters to create a RWH system

    chart_wspd_wdir - show wind speed vs. wind direction
    chart_wspd_wdir_monthly - show wind speed vs. wind direction for each month separate
    chart_windrose - show the wind rose for all time
    chart_windrose_yearly - show the wind rose for each year separate
    chart_ym_heatmap - create a heatmap from two given fields from the DataFrame on the year month grouping
    chart_freq - show a histogram for a particular defined field
    chart_precip_sum_ym - show the precipitation total per year-month
    chart_precip_sum_yw - show the precipitation total per year-week
    chart_precip_sum_to_h_yearly - show the hours of precipitation per month for every year separate
    chart_precip_month - show the categorized precipitation data per month
    chart_df_totals - print the table with the totals

return:
    all charts are saved as *.png files and can be find in the diagrams folder

"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from yenikas_weather.scripts import calc_rwh_system as rwh_system
from yenikas_weather.scripts import setup_environment as env


"""Show parameters"""
env.set_pd_environments()
# constants
config = env.get_config()
"""Parameters"""
cat_month = pd.CategoricalDtype(eval(config['rwh']['month_order']), ordered=True)

PROJECT_ROOT = env.get_project_root()
file_path = "{}/{}".format(PROJECT_ROOT, config['files']['weatherFile'])
diagram_path = "{}/diagrams/".format(PROJECT_ROOT)

def chart_wspd_wdir(df_tmp, filename):
    """wind speed vs wind direction chart total"""
    df_tmp.reset_index(drop=False, inplace=True)
    df_wdir_wspd \
        = df_tmp.pivot_table(index='wspd_5', columns='wdir_10', values='yyyy_mm', aggfunc='count')
    df_wdir_wspd.sort_index(level=0, ascending=False, inplace=True)

    fig1, (ax1) = plt.subplots(1, 1, figsize=(20, 12))
    fig1.suptitle('Wind data wind direction to wind speed')

    sns.heatmap(df_wdir_wspd, ax=ax1, cmap="YlGnBu", annot=True, linewidth=1, square=True, fmt='g')
    # ax1.set_title("Continent correlation")
    plt.xlabel('wind direction °')
    plt.ylabel('wind speed kph')
    fig1.savefig(diagram_path+"heatmap_" + filename + ".png")
    plt.close()


def chart_wspd_wdir_monthly(df_tmp, filename):
    """wind speed vs wind direction chart per month"""
    df_tmp.reset_index(drop=False, inplace=True)
    for m in range(1, 12):
        df_tmp_month = df_tmp[df_tmp["month"] == m]
        df_wdir_wspd \
            = df_tmp_month.pivot_table(index='wspd_5', columns='wdir_10', values='yyyy_mm', aggfunc='count')
        df_wdir_wspd.sort_index(level=0, ascending=False, inplace=True)

        fig1, (ax1) = plt.subplots(1, 1, figsize=(20, 12))
        fig1.suptitle('Wind data wind direction to wind speed for month '+str(m))
        sns.heatmap(df_wdir_wspd, ax=ax1, cmap="YlGnBu", annot=True, linewidth=1, square=True, fmt='g')
        # ax1.set_title("Continent correlation")
        plt.xlabel('wind direction °')
        plt.ylabel('wind speed kph')
        fig1.savefig(diagram_path+"monthly/heatmap_" +filename+ "_" +str(m)+ ".png")
        plt.close()
        del df_tmp_month
        del df_wdir_wspd


def chart_windrose(df_tmp):
    """wind rose with wind speed and direction"""
    from windrose import WindroseAxes

    df_wind = df_tmp[['wdir','wspd']]
    fig3 = plt.figure(figsize=(7, 7), dpi=80, facecolor='w', edgecolor='w')
    rect = [0.15, 0.15, 0.7, 0.7]
    ax3 = WindroseAxes(fig3, rect)
    fig3.add_axes(ax3)
    ax3.bar(df_wind["wdir"], df_wind["wspd"], normed=True, opening=1)
    ax3.set_title('wind direction with speed')
    plt.savefig(diagram_path+"windrose_wdir" + "" + ".png")
    plt.close()


def chart_windrose_yearly(df_tmp):
    """wind rose with wind speed and direction separate for each year"""
    from windrose import WindroseAxes

    # Create figure and plot space
#    df_tmp.reset_index(inplace=True)
    years = df_tmp['year'].unique()
    for y in years:
        df_wind = df_tmp[df_tmp["year"] == y]
        df_wind = df_wind[['wdir', 'wspd']]
        fig3 = plt.figure(figsize=(7, 7), dpi=80, facecolor='w', edgecolor='w')
        rect = [0.15, 0.15, 0.7, 0.7]
        ax3 = WindroseAxes(fig3, rect)
        fig3.add_axes(ax3)
        ax3.bar(df_wind["wdir"], df_wind["wspd"], normed=True, opening=1)
        ax3.set_title('wind direction with speed for year ' +str(y))
        plt.savefig(diagram_path+"yearly/windrose_wdir" + str(y) + ".png")
        plt.close()


def chart_ym_heatmap(df_tmp, fields, filename, title, xlabel, ylabel, cmap):
    """heatmap chart that shows the 3 entered fields
            please enter a dataFrame with exactly 3 fields"""
    # df_tmp.reset_index(drop=False, inplace=True)
    df_pivot = df_tmp.pivot(fields[0], fields[1], fields[2])
    fig2, (ax2) = plt.subplots(1, 1, figsize=(20, 12))
    fig2.suptitle(title)
    sns.heatmap(df_pivot, ax=ax2, cmap=cmap, annot=True, fmt='g')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig2.savefig(diagram_path+"heatmap_" + filename + ".png")
    plt.close()
    del df_pivot


def chart_freq(df_tmp, filename, title, xlabel, ylabel, bins=50):
    """histogram of the ne field of the dataFrame """
    f = plt.figure(figsize=(20, 12))
    f = sns.displot(df_tmp)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(diagram_path+"distplot_" + filename + ".png")
    plt.close()


def chart_precip_sum_ym(df_tmp, file_name):
    """lineplot precip_sum per year-month"""
    import datetime
    # Create figure and plot space
    df_tmp.reset_index(inplace=True)
    df_tmp = df_tmp[df_tmp['year'] >= datetime.datetime.today().year -7]
    fig1, (ax1) = plt.subplots(1, 1, figsize=(15, 8))
    fig1.suptitle('Year-Month charts')
    sns.set(style="darkgrid"
            , rc={"lines.linewidth": 1, "legend.fontsize": 8, "legend.loc": 'upper center'})
    sns.lineplot(data=df_tmp, sort=False,
                 x='month', y="precip_sum",
                 hue="year",
                 # size="precip_sum",
                 legend='full', palette='Paired', markers=True,
                 )
    ax1.set(xticks=df.month.values)
    ax1.set_xlabel("Year-Month")
    ax1.set_ylabel("recipation in mm", labelpad=10)
    ax1.set_title("Year-Month precipation sum")
    fig1.savefig(diagram_path+"lineplot"+file_name+".png")
    plt.close()


def chart_precip_sum_yw(df_tmp):
    """lineplot precip_sum per year-week"""
    import datetime
    # Create figure and plot space
#    df_tmp.reset_index(inplace=True)
    df_tmp = df_tmp[df_tmp['year'] >= datetime.datetime.today().year -7]
    df_tmp_grp = df_tmp.groupby(['year','week'], as_index=True, sort=True) \
        .agg(precip_sum=("precip", "sum")
             , precip_std=("precip", np.std)
             )
    fig1, (ax1) = plt.subplots(1, 1, figsize=(15, 8))
    fig1.suptitle('Year-Week charts')
    sns.set(style="darkgrid"
            , rc={"lines.linewidth": 1, "legend.fontsize": 8, "legend.loc": 'upper center'})
    sns.lineplot(data=df_tmp_grp, sort=False,
                 x='week', y="precip_sum",
                 hue="year",
                 # size="precip_std", #style="precip_std",
                 legend='full', palette='Paired', markers=True,
                 )
    ax1.set(xticks=df.week.values)
    ax1.set_xlabel("Year-week")
    ax1.set_ylabel("recipation in mm", labelpad=10)
    ax1.set_title("Year-week precipation sum")
    fig1.savefig(diagram_path+"lineplot_yw_precip_sum.png")
    plt.close()


def chart_precip_sum_to_h_yearly(df_tmp, df_tmp_grp):
    """pairplot of precip_sum, precip_h_sum and stored_grp_min separate for each year"""
    df_tmp_grp.reset_index(inplace=True)
    years = df_tmp['year'].unique()
    for y in years:
        df_grp_tmp = df_tmp_grp[df_tmp_grp["year"] == y]
        df_grp_tmp.set_index(['year', 'month'], inplace=True)
        g1 = sns.pairplot(df_grp_tmp[['precip_sum', 'precip_h_sum', 'stored_grp_min']]
                          , hue="stored_grp_min", diag_kind='kde', plot_kws={"s": 8})
        plt.suptitle('Year-Month precipation sum to hours')
        g1.savefig(diagram_path+"yearly/pairplot_precip_sum_h" + str(y) + ".png")
        plt.close()


def chart_precip_month(df_tmp, filename):
    """heatmap of the categorized precipitation per year-month"""
#    df_tmp.reset_index(drop=False, inplace=True)
    df_tmp = df_tmp[df_tmp['precip_grp'] != '00']
    df_precip_month \
        = df_tmp.pivot_table(index='precip_grp', columns='month', values='yyyy_mm', aggfunc='count')
    df_precip_month.sort_index(level=0, ascending=False, inplace=True)
    fig1, (ax1) = plt.subplots(1, 1, figsize=(20, 12))
    fig1.suptitle('rain fall per month')
    sns.heatmap(df_precip_month, ax=ax1, cmap="YlGnBu", annot=True, linewidth=1, square=True, fmt='g')
    # ax1.set_title("Continent correlation")
    plt.xlabel('month')
    plt.ylabel('rain fall in mm grouped')
    fig1.savefig(diagram_path+"heatmap_" + filename + ".png")
    plt.close()


def chart_df_totals(df_tmp_grp):
    """plot table with totls of the DataFrame"""
    f = plt.figure(figsize=(20, 12))
    ax = f.add_subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    tab = pd.plotting.table(ax, df_tmp_grp, loc='upper right')
    # tab.auto_set_font_size(False)
    # tab.set_fontsize(8)
    # tab.scale(1.2, 1.2)
    plt.savefig(diagram_path+"df_totals.png", transparent=True)
    plt.close()


"""get the data"""
df, df_total, df_ym, df_y = rwh_system.main()


"""plot data"""
sns.set_theme()
axis_font = {'fontname': 'Arial', 'size': '6'}

chart_df_totals(df_total)

chart_wspd_wdir(df, "wdir_wspd")
chart_wspd_wdir_monthly(df, "wdir_wspd")

chart_windrose(df)
chart_windrose_yearly(df)

chart_precip_month(df, "precip_month")

chart_freq(df_ym["precip_sum"], "precip_freq_ym", 'Frequency of percipation (mm/yyyy-mm)'
    , 'percipation in mm grouped', "#", 50)
chart_freq(df_y["precip_sum"], "precip_freq_y", 'Frequency of percipation (yyyy)'
    , 'percipation in mm grouped', "#", 50)
chart_freq(df["wspd"], "wspd", 'Frequency of speed of the wind (kph/yyyy-mm)'
    , 'wind speed in kph', "#", 50)
chart_freq(df["wdir"], "wdir", 'Frequency of the wind direction (°/yyyy-mm)'
    , 'wind direction in °', "#", 50)
chart_precip_sum_ym(df_ym, "_ym_precip_sum")
#chart_precip_sum_ym(df_y, "_y_precip_sum")
chart_precip_sum_yw(df)
# chart_precip_sum_to_h(df, df_ym)


df_rsm, df_rsm_total = rwh_system.group_rwh_data_ym(df, ['rain_season', 'month'], False)

df_rsm.reset_index(drop=False, inplace=True)
df_rsm['month'] = df_rsm['month'].astype(cat_month)
# df = df.sort_values(['rain_season', 'month'])
chart_ym_heatmap(df_rsm, ["rain_season", "month", "precip_sum"]
    , "precip_sum_ym", "amount of rain in mm/m² ", "month", "rain season", "YlGnBu")
chart_ym_heatmap(df_rsm, ["rain_season", "month", "precip_h_sum"]
    , "precip_h_sum_ym", "hours of rain ", "month", "rain season", "YlGnBu")
chart_ym_heatmap(df_rsm, ["rain_season", "month", "precip_h_avg"]
    , "precip_h_avg_ym", "average hours of rain per day", "month", "rain season", "YlGnBu")
chart_ym_heatmap(df_rsm, ["rain_season", "month", "precip_h_med"]
    , "precip_h_med_ym", "median hours of rain per day", "month", "rain season", "YlGnBu")

chart_ym_heatmap(df_rsm, ["rain_season", "month", "dry_day_sum"]
    , "dry_day_sum_ym", "# of days without rain ", "month", "rain season", "coolwarm")
chart_ym_heatmap(df_rsm, ["rain_season", "month", "wet_day_sum"]
    , "wet_day_sum_ym", "# of days with rain ", "month", "rain season", "YlGnBu")

chart_ym_heatmap(df_rsm, ["rain_season", "month", "precip_mm_h_avg"]
    , "precip_mm_h_avg_ym", "average rainfall in mm/h ", "month", "rain season", "coolwarm")
cmap = LinearSegmentedColormap.from_list('ryg', ["white", "lightblue", "blue", "lavender", "orange", "orangered"\
    , "red", "maroon", "plum", "violet", "purple"], N=1024)
chart_ym_heatmap(df_rsm, ["rain_season", "month", "precip_mm_h_max"]
    , "precip_mm_h_max_ym", "maximum rainfall in mm/h ", "month", "rain season", cmap)

cmap = LinearSegmentedColormap.from_list('ryg', ["darkred", "yellow", "lightyellow"
    , "yellowgreen", "lightgreen", "palegreen", "green", "darkgreen", "fuchsia", "violet", "purple"], N=1024)
chart_ym_heatmap(df_rsm, ["rain_season", "month", "collected_sum"]
    , "collected_sum_ym", "amount of rain collected in l ", "month", "rain season", cmap)
chart_ym_heatmap(df_rsm, ["rain_season", "month", "net_collected_sum"]
    , "net_collected_sum_ym", "(net)amount of rain collected in l ", "month", "rain season", cmap)
chart_ym_heatmap(df_rsm, ["rain_season", "month", "overrun_sum"]
    , "overrun_sum_ym", "total amount of rain that couldnt be stored in l ", "month", "rain season", "YlGnBu")
chart_ym_heatmap(df_rsm, ["rain_season", "month", "net_overrun_sum"]
    , "net_overrun_sum_ym", "(net)amount of rain that didnt reach the tank in l ", "month", "rain season", "YlGnBu")
chart_ym_heatmap(df_rsm, ["rain_season", "month", "tank_overrun_sum"]
    , "tank_overrun_sum_ym", "(net)amount of rain that couldnt be stored in l ", "month", "rain season", "YlGnBu")

chart_ym_heatmap(df_rsm, ["rain_season", "month", "person_sum"]
    , "person_sum_ym", " water used by persons in l ", "month", "rain season", "RdYlGn")
chart_ym_heatmap(df_rsm, ["rain_season", "month", "garden_sum"]
    , "garden_sum_ym", " water used for the garden in l ", "month", "rain season", "RdYlGn")

chart_ym_heatmap(df_rsm, ["rain_season", "month", "stored_min"]
    , "stored_min_ym", "minimum fill state of the water storage in l ", "month", "rain season", "RdYlGn")
chart_ym_heatmap(df_rsm, ["rain_season", "month", "stored_max"]
    , "stored_max_ym", "maximum fill state of the water storage in l ", "month", "rain season", "RdYlGn")
chart_ym_heatmap(df_rsm, ["rain_season", "month", "tank_empty_sum"]
    , "tank_empty_sum_ym", "# of days the water storage is empty ", "month", "rain season", "coolwarm")
chart_ym_heatmap(df_rsm, ["rain_season", "month", "tank_low_sum"]
    , "tank_low_sum_ym", "# of days the water storage is below 10% ", "month", "rain season", "coolwarm")

df_ym.reset_index(drop=False, inplace=True)
chart_ym_heatmap(df_ym, ["year", "month", "temp_min"]
    , "temp_min_ym", "minimum temperature in °C ", "month", "year", "coolwarm")
chart_ym_heatmap(df_ym, ["year", "month", "temp_avg"]
    , "temp_avg_ym", "average temperature in °C ", "month", "year", "coolwarm")
chart_ym_heatmap(df_ym, ["year", "month", "temp_max"]
    , "temp_max_ym", "maximum temperature in °C ", "month", "year", "coolwarm")

chart_ym_heatmap(df_ym, ["year", "month", "humidity_min"]
    , "humidity_min_ym", "minimum humidity in % ", "month", "year", "RdYlGn")
chart_ym_heatmap(df_ym, ["year", "month", "humidity_avg"]
    , "humidity_avg_ym", "average humidity in % ", "month", "year", "RdYlGn")
chart_ym_heatmap(df_ym, ["year", "month", "humidity_max"]
    , "humidity_max_ym", "maximum humidity in % ", "month", "year", "RdYlGn")


