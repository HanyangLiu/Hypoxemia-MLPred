import sys

if sys.version_info >= (3, 0):
    import configparser
    config = configparser.ConfigParser()
else:
    import ConfigParser
    config = ConfigParser.ConfigParser()

config.read('file_config/data.conf')
