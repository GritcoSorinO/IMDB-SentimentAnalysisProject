import csv
import pandas as pd

def obtain_data_frame_from_file_path(file_path):

    tsv_file = open(file_path, encoding='utf-8')
    read_tsv = csv.reader(tsv_file, delimiter='\t')
    df_list = list()

    for row in read_tsv:
        df_list.append(row)
    df = pd.DataFrame(df_list[1:], columns=df_list[0])

    return df
