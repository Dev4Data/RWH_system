import pandas as pd
import configparser
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]


def get_config():
    config = configparser.ConfigParser()
    config_file = "{}\settings\settings.ini".format(PROJECT_ROOT)
    key_file = "{}\settings\key.ini".format(PROJECT_ROOT)
    config.read([config_file, key_file])
    return config


def set_pd_environments():
    config = get_config()
    """setup pandas output environment"""
    pd.set_option('max_colwidth', eval(config.get('DEFAULT', 'displayMaxColWidth', fallback=None)))
    pd.set_option('max_seq_item', eval(config.get('DEFAULT', 'displayMaxSeqItem', fallback=None)))
    pd.set_option('display.width', int(config['DEFAULT']['displayWidth']))
    pd.set_option('display.max_columns', eval(config.get('DEFAULT', 'displayMaxColumns', fallback=None)))

