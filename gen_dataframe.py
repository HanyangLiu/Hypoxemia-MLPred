# generate DataFrame for static and real-time data

import pandas as pd
from utils.utility_parsing_raw import ParseDynamicData, ParseStaticData, ParseICD
from file_config.config import config

# path
vitals_dir = config.get('data', 'vitals_dir')
demographic_file = config.get('data', 'demographic_file')
icd_file = config.get('data', 'icd_file')
df_static_file = config.get('processed', 'df_static_file')
df_dynamic_file = config.get('processed', 'df_dynamic_file')


def gen_static_dataframe():

    # parse demographic data into DataFrame
    print('Start parsing demographic...')
    static_parser = ParseStaticData(demographic_file)
    df_static = static_parser.gen_static_dataframe()

    # parse ICD data into DataFrame, lines with PatientID not in demographic are removed
    print('Start parsing ICD...')
    id_converter = static_parser.id_converter
    icd_parser = ParseICD(id_converter, icd_file)
    df_icd = icd_parser.parse_icd()

    # merge ICD data to demographic DataFrame
    print('Merging...')
    df = pd.merge(df_static, df_icd, how='outer', on='pid')
    df.to_csv(df_static_file, index=False)
    print('Finished generating static DataFrame!')


def gen_realtime_dataframe():

    # initiate parsers
    static_parser = ParseStaticData(demographic_file=demographic_file)
    dynamic_parser = ParseDynamicData(vitals_dir)
    id_converter = static_parser.id_converter  # PatientID to pid

    # Parsing vital files into DataFrame
    print('Start parsing vitals...')
    anesthesia_start_time = static_parser.start_time_dict
    df = dynamic_parser.parse_vitals(start_time_dict=anesthesia_start_time, id_converter=id_converter)
    df.to_csv(df_dynamic_file, index=False)  # save DataFrame to csv
    print('Finished generating real-time DataFrame!')


if __name__ == '__main__':
    gen_static_dataframe()
    gen_realtime_dataframe()




