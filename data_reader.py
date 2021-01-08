from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from random import randint
import json
import pandas as pd
from utils import *
from sklearn import preprocessing
import nltk

tokenizer=nltk.WordPunctTokenizer()
stop_words=nltk.corpus.stopwords.words('english')

def load_pretrained_wv(path):
    wv = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            items = line.split(' ')
            wv[items[0]] = torch.DoubleTensor([float(a) for a in items[1:]])
    return wv

def pad_or_crop(field,max_tokens,to_add):
    if len(field)>max_tokens:
        field=field[0:max_tokens]
    if len(field)<max_tokens:
        for i in range(max_tokens-len(field)):
            field+=to_add
    return field


def pad_or_crop_with_rep(field,max_tokens,dictt,field_type):

    final=[]

    for f in field:
        if f in dictt.keys():
            final.append(f)
        else:
            final.append('unk')

    if len(final)>max_tokens:
        final=final[0:max_tokens]
    if len(final)<max_tokens:
        inter=final.copy()
        if len(inter)==0:
            #print('here empty')
            if field_type in ['description','attributes']:
                inter=[',']
            else:
                inter = ['.']
        j=0
        for i in range(max_tokens-len(final)):
            final.append(inter[j%len(inter)])
            j+=1

    return final

def encode_field(field,dictt,field_type):
    vect_ = []
    for qu in field:
        try:
            vect_ += [dictt[qu]]
        except:

            if field_type in ['description','attributes']:
                vect_ += [dictt[',']]
            else:
                vect_ += [dictt['.']]

    return vect_


def get_www_all_features(feature_file):
    ids_left = []
    ids_right = []
    features = []
    labels  = []
    f_f = open(feature_file,'r')
    line = f_f.readline()
    for line in f_f:
        seps = line.strip().split(',')
        qid = seps[0]
        tid = seps[2]
        ids_left.append(qid)
        ids_right.append(tid)
        rel = seps[-1]
        labels.append(int(rel))
        '''
        if int(rel) > 0:
            labels.append(1)
        else:
            labels.append(0)
        '''
        #feat_range=np.arange(3,25)
        q_doc_f = np.array([float(each) for each in seps[3:-1]])
        #q_doc_f = np.array([float(each) for num,each in enumerate(seps) if num in feat_range or num==41])
        #feat_range = np.arange(25, 41)
        #q_doc_f = np.array([float(each) for num, each in enumerate(seps) if num in feat_range])
        features.append(q_doc_f)


    df = pd.DataFrame({
        'id_left': ids_left,
        'id_right': ids_right,
        'features': features,
        'label': labels
    })
    return df


class DataAndQuery(Dataset):
    def __init__(self,file_name, Train,wv,word_to_index,index_to_word,output_file,args):

        if not Train:
            self.wv = wv
            self.word_to_index = word_to_index
            self.index_to_word = index_to_word
        else:
            self.word_to_index = {}
            self.index_to_word = []

            pretrained_wv = args.word_embedding
            # pretrained_wv = '/home/mohamedt/ARCI/glove.6B.300d.txt'
            self.wv = load_pretrained_wv(pretrained_wv)
            for i, key in enumerate(self.wv.keys()):
                self.word_to_index[key] = i
                self.index_to_word.append(key)

        max_tokens_desc = args.max_meta_len
        max_tokens_att = args.max_attributes_len
        max_tokens_query = args.max_query_len
        max_values_struct = args.max_values_len

        all_desc = []
        all_att = []
        all_values = []
        all_query = []
        labels = []
        all_values_struct = []

        path = os.path.join(args.data_path,args.values_file)

        with open(path) as f:
            dt = json.load(f)

        data_csv = pd.read_csv(os.path.join(args.data_path,'features2.csv'))

        test_data = data_csv['table_id']
        query = data_csv['query']

        text_file = open(file_name, "r")
        lines = text_file.readlines()

        queries_id = []
        list_lines = []

        for line in lines:
            # print(line)
            line = line[0:len(line) - 1]
            aa = line.split('\t')
            queries_id += [aa[0]]
            list_lines.append(aa)

        queries_id = [int(i) for i in queries_id]

        qq = np.sort(list(set(queries_id)))

        test_data = list(test_data)

        to_save=[]
        all_query_labels=[]
        all_semantic=[]
        normalize=True

        all_df = get_www_all_features(os.path.join(args.data_path,'features2.csv'))

        for q in qq:
            #print(q)
            # if q>2:
            #     break
            indexes = [i for i, x in enumerate(queries_id) if x == q]
            indices = data_csv[data_csv['query_id'] == q].index.tolist()
            #print(indexes)

            inter = np.array(list_lines)[indexes]

            test_query = list(query[indices])[0]

            #query_tokens = preprocess(test_query, 'description')
            query_tokens = preprocess_seq(test_query, tokenizer, stop_words)
            query_tokens = query_tokens.split(' ')
            query_ = pad_or_crop_with_rep(query_tokens, max_tokens_query, self.word_to_index, 'query')
            vector_query = [encode_field(query_, self.word_to_index, 'query')]

            for item in inter:
                if item[2] in test_data:

                    all_query_labels.append(q)

                    rel = float(data_csv[((data_csv['query_id'] == q) & (data_csv['table_id'] == item[2]))].iloc[0]['rel'])
                    table = dt[item[2]]

                    pgTitle_feat = table['pgTitle']
                    if len(pgTitle_feat) > 0:
                        pgTitle_feat = pgTitle_feat.split(' ')
                    else:
                        pgTitle_feat = []

                    secondTitle_feat = table['secondTitle']
                    if len(secondTitle_feat) > 0:
                        secondTitle_feat = secondTitle_feat.split(' ')
                    else:
                        secondTitle_feat = []

                    caption_feat = table['caption']
                    if len(caption_feat) > 0:
                        caption_feat = caption_feat.split(' ')
                    else:
                        caption_feat = []

                    description=pgTitle_feat+secondTitle_feat+caption_feat

                    original_attributes = table['attributes']
                    if len(original_attributes)>0:
                        original_attributes = original_attributes.split(' ')
                    else:
                        original_attributes=[]

                    values = table['data']
                    if len(original_attributes) > 0:
                        values = values.split(' ')
                    else:
                        values = []

                    description = pad_or_crop_with_rep(description, max_tokens_desc, self.word_to_index, 'description')
                    original_attributes = pad_or_crop_with_rep(original_attributes, max_tokens_att, self.word_to_index,
                                                               'attributes')
                    values = pad_or_crop_with_rep(values, max_tokens_desc, self.word_to_index, 'description')

                    values_struct = table['data_struct']

                    cols = []
                    rows = []
                    table_values_struct = []
                    if len(values_struct) > 0:
                        values_struct = np.array(values_struct)
                        for i in range(values_struct.shape[0]):
                            rows.append(values_struct[i, :])

                        for i in range(values_struct.shape[1]):
                            cols.append(values_struct[:, i])

                        for row in rows:
                            row = [x for r in row for x in r.split(' ') if len(x) > 0]
                            vector_row = encode_field(row, self.word_to_index, 'attributes')
                            if len(vector_row) == 0:
                                continue
                            table_values_struct.append(vector_row)

                        for col in cols:
                            col = [x for c in col for x in c.split(' ') if len(x) > 0]
                            vector_col = encode_field(col, self.word_to_index, 'attributes')
                            if len(vector_col) == 0:
                                continue

                            table_values_struct.append(vector_col)

                    if len(table_values_struct) > max_values_struct:
                        table_values_struct = table_values_struct[:max_values_struct]
                    else:
                        for i in range(max_values_struct - len(table_values_struct)):
                            table_values_struct.append([self.word_to_index[',']])


                    vector_desc = [encode_field(description, self.word_to_index, 'attributes')]
                    vector_att = [encode_field(original_attributes, self.word_to_index, 'description')]
                    vector_values = [encode_field(values, self.word_to_index, 'description')]

                    el=all_df.loc[(all_df['id_left'] == str(q)) & (all_df['id_right'] == item[2])]
                    el=el['features']
                    all_semantic.append(list(el.values)[0])


                    all_desc.append([vector_desc])
                    all_att.append([vector_att])
                    all_query.append([vector_query])
                    all_values.append([vector_values])
                    all_values_struct.append(table_values_struct)

                    labels.append(rel)
                    to_save.append(item)

        all_semantic = np.stack(all_semantic, axis=0)
        if normalize:
            #scaler = preprocessing.StandardScaler()
            #all_semantic = scaler.fit_transform(all_semantic)
            all_semantic=preprocessing.normalize(all_semantic)

        self.all_desc = all_desc
        self.all_desc = torch.tensor(self.all_desc)
        self.all_att = all_att
        self.all_att = torch.tensor(self.all_att)
        self.all_values = all_values
        self.all_values = torch.tensor(self.all_values)
        self.all_query = all_query
        self.all_query = torch.tensor(self.all_query)

        self.all_query_labels=all_query_labels
        self.all_semantic=all_semantic
        self.all_semantic=torch.tensor(self.all_semantic)

        self.all_values_struct = all_values_struct

        self.labels = labels
        inter = np.array(to_save)
        np.savetxt(output_file, inter, fmt="%s", delimiter='\t')





    def __getitem__(self, t):
        """
            return: the t-th (center, context) word pair and their co-occurrence frequency.
        """
        ## Your codes go here
        return self.all_desc[t],self.all_att[t],self.all_query[t],self.labels[t],self.all_semantic[t],self.all_values[t]

    def __len__(self):
        """
            return: the total number of (center, context) word pairs.
        """
        ## Your codes go here
        return len(self.all_desc)
