import numpy as np
import json
import os
from random import shuffle
import pandas as pd
from tqdm import tqdm
from utils import *
import nltk
import argparse

parser = argparse.ArgumentParser(description='PreprocessWikiTables', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--tables_path', default='/home/mohamed/PycharmProjects/Data-Search-Project/tables_redi2_1')
parser.add_argument('--collection_path', default='wikiTables/features2.csv')
parser.add_argument('--output_folder', default='wikiTables')
parser.add_argument('--output_file', default='data_fields_with_struct_values2.json')

args = parser.parse_args()

data_folder = args.tables_path
data_csv=pd.read_csv(args.collection_path)

tokenizer=nltk.WordPunctTokenizer()
stop_words=nltk.corpus.stopwords.words('english')

def prepare_table(test_table):
    attributes = test_table['title']
    pgTitle = test_table['pgTitle']
    secondTitle = test_table['secondTitle']
    caption = test_table['caption']
    data = test_table['data']

    pgTitle_feat = preprocess_seq(pgTitle, tokenizer,stop_words)
    secondTitle_feat = preprocess_seq(secondTitle, tokenizer,stop_words)
    caption_feat = preprocess_seq(caption, tokenizer,stop_words)

    #pgTitle_feat=' '.join(pgTitle_feat)
    #secondTitle_feat = ' '.join(secondTitle_feat)
    #caption_feat = ' '.join(caption_feat)

    data_csv = pd.DataFrame(data, columns=attributes)
    attributes = list(data_csv)

    inter_att = ' '.join(attributes)
    all_att_tokens = preprocess_seq(inter_att, tokenizer,stop_words)

    if len(all_att_tokens) == 0:
        data_csv = data_csv.transpose()
        # vec_att = np.array(attributes).reshape(-1, 1)
        data_csv_array = np.array(data_csv)
        # data_csv_array = np.concatenate([vec_att, data_csv_array], axis=1)
        if data_csv_array.size > 0:
            attributes = data_csv_array[0, :]
            inter_att = ' '.join(attributes)
            all_att_tokens = preprocess_seq(inter_att, tokenizer, stop_words)
            data_csv = pd.DataFrame(data_csv_array, columns=attributes)

            data_csv = data_csv.drop([0], axis=0).reset_index(drop=True)
        else:
            data_csv = data_csv.transpose()

    # all_att_tokens = []
    # for att in attributes:
    #     att_tokens = preprocess_seq(att, tokenizer,stop_words)
    #     all_att_tokens += att_tokens



    original_attributes = all_att_tokens
    values = data_csv.values
    #original_attributes = ' '.join(original_attributes)

    list_value=[[] for _ in range(values.shape[0])]
    for row,val in enumerate(values):
        for col,val_col in enumerate(val):
            val_col=preprocess_seq(val_col,tokenizer,stop_words)
            list_value[row].append(val_col)

    #list_values=np.array(list_value)

    data = ' '.join(y for x in values for y in x)
    data_tokens = preprocess_seq(data,tokenizer,stop_words)
    #data_tokens = ' '.join(data_tokens)

    return pgTitle_feat,secondTitle_feat,caption_feat,original_attributes,data_tokens,list_value



# list_of_categories = os.listdir(data_folder)
# nb_files = len(list_of_categories)
# mylist = list(range(nb_files))
# #print("start shuffle")
# shuffle(mylist)
# #print("shuffle done")
# list_of_categories = np.array(list_of_categories)[mylist]
# nb_files_in_training = nb_files
# nb_tables_per_file=400
# list_of_categories = list_of_categories[:nb_files_in_training]

test_data=data_csv['table_id']

docs={}
with tqdm(total=len(test_data)) as pbar0:
    for jj, table in enumerate(test_data):
        inter = table.split("-")
        file_number = inter[1]
        table_number = inter[2]

        file_name = 're_tables-' + file_number + '.json'

        table_name = 'table-' + file_number + '-' + table_number

        path = os.path.join(data_folder, file_name)

        with open(path) as f:
            dt = json.load(f)

        test_table = dt[table_name]

        pgTitle_feat, secondTitle_feat, caption_feat, original_attributes, data_tokens, list_values = prepare_table(test_table)

        if table_name not in docs:
            docs[table_name] = {}
            docs[table_name]['attributes'] = original_attributes
            docs[table_name]['pgTitle'] = pgTitle_feat
            docs[table_name]['secondTitle'] = secondTitle_feat
            docs[table_name]['caption'] = caption_feat
            docs[table_name]['data'] = data_tokens
            docs[table_name]['data_struct'] = list_values


        pbar0.update(1)

json_file=os.path.join(args.output_folder,args.output_file)
with open(json_file,'w') as outfile:
    json.dump(docs, outfile)