"""global Setup of the environment
author: Matthis (Dev4Data-github@online.ms)
functions:
    get_config - get the settings from the *.ini file(s)
    set_pd_environments -

"""
import pandas as pd
import configparser
from pathlib import Path


def get_project_root():
    return Path(__file__).parents[1]


def get_config():
    """method to get the configuration parameters
            the parameters can be saved in multiple files

        parameters:
        return:
            ConfigParser object
    """
    PROJECT_ROOT = get_project_root()
    config = configparser.ConfigParser()
    config_file = "{}\settings\settings.ini".format(PROJECT_ROOT)
    key_file = "{}\settings\key.ini".format(PROJECT_ROOT)
    config.read([config_file, key_file])
    return config


def set_pd_environments():
    """method to set environmental output parametes
            how output tables are displayed """
    config = get_config()
    """setup pandas output environment"""
    pd.set_option('max_colwidth', eval(config.get('DEFAULT', 'displayMaxColWidth', fallback=None)))
    pd.set_option('max_seq_item', eval(config.get('DEFAULT', 'displayMaxSeqItem', fallback=None)))
    pd.set_option('display.width', int(config['DEFAULT']['displayWidth']))
    pd.set_option('display.max_columns', eval(config.get('DEFAULT', 'displayMaxColumns', fallback=None)))
    pd.set_option('display.max_rows', eval(config.get('DEFAULT', 'displayMaxRows', fallback=None)))

