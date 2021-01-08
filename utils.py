import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import re
import torch
import pandas as pd
import subprocess
import torch.nn.functional as F


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

def preprocess_seq(input,wpt,stop_words):

    #w = re.sub(r'[^a-zA-Z0-9@$%\s]', ' ', input, re.I | re.A)
    w=input
    w = w.strip()
    # tokenize document
    tokens = wpt.tokenize(w)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]

    camel_tokens=[]

    for w in filtered_tokens:
        inter = camel_case_split(w)
        camel_tokens += inter

    tokens=camel_tokens

    # convert to lower case
    tokens = ' '.join(tokens)
    tokens = tokens.lower()

    #tokens=tokens.split(' ')



    return tokens

def preprocess(input,type):

    if type=='attribute':
        w = input.replace('-', " ").replace('_', ' ').replace('/', ' ').replace(',', ' ').replace('.', ' ').replace('|', ' ').replace(':', ' ')
        #w = input.replace('_', ' ')

        tokens = word_tokenize(w)

        camel_tokens=[]

        for w in tokens:
            inter = camel_case_split(w)
            camel_tokens += inter

        tokens=camel_tokens

        # convert to lower case
        tokens = [w.lower() for w in tokens]

        inter_words = []
        for w in tokens:
            inter = re.sub(r'\u2013+', ' ', w).split()
            inter_words += inter

        inter_words2 = []
        for w in inter_words:
            inter = re.sub(r'\u2014+', ' ', w).split()
            inter_words2 += inter
        # remove punctuation from each word
        # table = str.maketrans('', '', string.punctuation)
        # stripped = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        words = [word for word in inter_words2 if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]

        # final_words = []
        # for w in words:
        #     inter = re.sub('([a-z])([A-Z])', r'\1 \2', w).split()
        #     final_words += inter

        final_words=words

        final_words = [tok for tok in final_words if isEnglish(tok)]

    elif type=='value':
        #w = input.replace('_', ' ').replace(',', ' ')
        w = input.replace('-', " ").replace('_', ' ').replace('/', ' ').replace(',', ' ').replace('.', ' ').replace('|', ' ').replace(':', ' ')
        #w=input

        tokens = word_tokenize(w)

        camel_tokens = []

        for w in tokens:
            inter = camel_case_split(w)
            camel_tokens += inter

        tokens = camel_tokens

        # convert to lower case
        tokens = [w.lower() for w in tokens]
        inter_words = []
        for w in tokens:
            inter = re.sub(r'\u2013+', ' ', w).split()
            inter_words += inter

        inter_words2 = []
        for w in inter_words:
            inter = re.sub(r'\u2014+', ' ', w).split()
            inter_words2 += inter

        # remove punctuation from each word
        # table = str.maketrans('', '', string.punctuation)
        # stripped = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic

        numerical_values=[]
        string_values=[]
        for word in inter_words2:
            try:
                float(word)
                numerical_values.append(word)

            except ValueError:
                string_values.append(word)


        string_values_final=[]
        for w in string_values:
            inter=re.split(r'(\d+)', w)

            for word in inter:
                if len(word)>0:
                    try:
                        float(word)
                        numerical_values.append(word)

                    except ValueError:
                        string_values_final.append(word)

        #keep 0 digits
        #numerical_values = [re.sub('\d', '#', s) for s in numerical_values]

        #keep 1 digit
        numerical_values_inter=[]
        for s in numerical_values:
            if s[0]=='-':
                ss=s[2::]
                ss=re.sub('\d', '#', ss)
                ss=s[0:2]+ss


            else:
                ss = s[1::]
                ss = re.sub('\d', '#', ss)
                ss = s[0] + ss

            numerical_values_inter += [ss]

        #keep 2 digits

        # for s in numerical_values:
        #     ss=s[2::]
        #     ss=re.sub('\d', '#', ss)
        #     ss=s[0:2]+ss
        #     numerical_values_inter+=[ss]

        numerical_values=numerical_values_inter
        inter_words2 = string_values_final

        words = [word for word in inter_words2 if word.isalpha() or word in['$','@','%','£','€','°']]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        stop_words.remove('d')
        stop_words.remove('m')
        stop_words.remove('s')

        words = [w for w in words if not w in stop_words]

        # final_words = []
        # for w in words:
        #     inter = re.sub('([a-z])([A-Z])', r'\1 \2', w).split()
        #     final_words += inter

        final_words=words

        final_words = [tok for tok in final_words if isEnglish(tok) or tok in['$','@','%','£','€','°']]

        final_words=final_words+numerical_values



    elif type=='value2':

        w = input.replace('-', " ").replace('_', ' ').replace('/', ' ').replace(',', ' ').replace('.', ' ').replace('|', ' ').replace(':', ' ')

        tokens = word_tokenize(w)

        # convert to lower case
        tokens = [w.lower() for w in tokens]
        inter_words = []
        for w in tokens:
            inter = re.sub(r'\u2013+', ' ', w).split()
            inter_words += inter

        inter_words2 = []
        for w in inter_words:
            inter = re.sub(r'\u2014+', ' ', w).split()
            inter_words2 += inter


        numerical_values=[]
        string_values=[]
        for word in inter_words2:
            try:
                float(word)
                numerical_values.append(word)

            except ValueError:
                string_values.append(word)


        string_values_final=[]
        for w in string_values:
            inter=re.split(r'(\d+)', w)

            for word in inter:
                if len(word)>0:
                    try:
                        float(word)
                        numerical_values.append(word)

                    except ValueError:
                        string_values_final.append(word)



        inter_words2 = string_values_final

        words = [word for word in inter_words2 if word.isalpha() or word in['$','@','%','£','€','°']]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        stop_words.remove('d')
        stop_words.remove('m')
        stop_words.remove('s')

        words = [w for w in words if not w in stop_words]

        final_words = []
        for w in words:
            inter = re.sub('([a-z])([A-Z])', r'\1 \2', w).split()
            final_words += inter

        final_words = [tok for tok in final_words if isEnglish(tok) or tok in['$','@','%','£','€','°']]

        final_words=final_words+numerical_values

    elif type == 'description':
        #w = input.replace('_', ' ').replace(',', ' ').replace('-', " ").replace('.', ' ')
        w = input.replace('-', " ").replace('_', ' ').replace('/', ' ').replace(',', ' ').replace('.', ' ').replace('|', ' ').replace(':', ' ')


        tokens = word_tokenize(w)

        camel_tokens = []

        for w in tokens:
            inter = camel_case_split(w)
            camel_tokens += inter

        tokens = camel_tokens

        # convert to lower case
        tokens = [w.lower() for w in tokens]
        inter_words = []
        for w in tokens:
            inter = re.sub(r'\u2013+', ' ', w).split()
            inter_words += inter

        inter_words2 = []
        for w in inter_words:
            inter = re.sub(r'\u2014+', ' ', w).split()
            inter_words2 += inter
        # remove punctuation from each word
        #table = str.maketrans('', '', string.punctuation)
        #stripped = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        words = [word for word in inter_words2 if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]

        # final_words=[]
        # for w in words:
        #     inter=re.sub('([a-z])([A-Z])', r'\1 \2', w).split()
        #     final_words+=inter

        final_words=words

        final_words = [tok for tok in final_words if isEnglish(tok)]

        not_to_use=['com','u','comma','separated','values','csv','data','dataset','https','api','www','http','non','gov','rows','p','download','downloads','file','files','p']

        final_words=[tok for tok in final_words if tok not in not_to_use]

    return final_words


def kernal_mus(n_kernels):

    l_mu = [1]
    if n_kernels == 1:
        return l_mu

    bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
    l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
    for i in range(1, n_kernels - 1):
        l_mu.append(l_mu[i] - bin_size)
    return l_mu


def kernel_sigmas(n_kernels):

    bin_size = 2.0 / (n_kernels - 1)
    l_sigma = [0.001]  # for exact match. small variance -> exact match
    if n_kernels == 1:
        return l_sigma

    l_sigma += [0.1] * (n_kernels - 1)
    return l_sigma

def load_checkpoint(model, optimizer, losslogger, filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        losslogger = checkpoint['losslogger']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, losslogger

def load_checkpoint_for_eval(model, filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model

def read_file_for_nfcg(file):
    text_file = open(file, "r")
    lines = text_file.readlines()

    queries_id = []
    list_lines = []

    for line in lines:
        # print(line)
        line = line[0:len(line) - 1]
        aa = line.split('\t')
        queries_id += [aa[0]]
        list_lines.append(aa)
    inter = np.array(list_lines)

    return inter

def calculate_metrics(inter, output_file,all_outputs,ndcg_file):
    inter2 = []

    for jj, item in enumerate(inter):
        item_inter = [i for i in item]
        item_inter[4] = str(all_outputs[jj])

        inter2.append(item_inter)

    inter3 = np.array(inter2)

    np.savetxt(output_file, inter3, fmt="%s")

    #batcmd = "./trec_eval -m ndcg_cut.5 "+ndcg_file+" " + output_file
    batcmd = "./trec_eval -m map " + ndcg_file + " " + output_file
    result = subprocess.check_output(batcmd, shell=True, encoding='cp437')
    res = result.split('\t')
    map = float(res[2])

    batcmd = "./trec_eval -m recip_rank " + ndcg_file + " " + output_file
    result = subprocess.check_output(batcmd, shell=True, encoding='cp437')
    res = result.split('\t')
    mrr = float(res[2])

    batcmd = "./trec_eval -m ndcg_cut.5 " + ndcg_file + " " + output_file
    result = subprocess.check_output(batcmd, shell=True, encoding='cp437')
    res = result.split('\t')
    ndcg = float(res[2])

    return ndcg,map,mrr




def calculate_ndcg(inter, output_file,all_outputs,ndcg_file):
    inter2 = []

    for jj, item in enumerate(inter):
        item_inter = [i for i in item]
        item_inter[4] = str(all_outputs[jj])

        inter2.append(item_inter)

    inter3 = np.array(inter2)

    np.savetxt(output_file, inter3, fmt="%s")

    batcmd = "./trec_eval -m ndcg_cut.5 "+ndcg_file+" " + output_file
    result = subprocess.check_output(batcmd, shell=True, encoding='cp437')
    res = result.split('\t')
    ndcg = float(res[2])

    return ndcg

def qrel_for_data(data,list_lines_qrels,output_file):
    #list_lines_qrels=np.array(list_lines_qrels)
    df = pd.DataFrame(list_lines_qrels)
    qrel_inter=[]
    for i in range(len(data)):
        row=data[i]
        ii=df[((df[0] == row[0]) & (df[2] == row[2]))]
        qrel_inter+=ii.values.tolist()

    qrel_inter=np.array(qrel_inter)

    np.savetxt(output_file, qrel_inter, fmt="%s",delimiter='\t')

def listnet_loss(y_i, z_i):
    """
    y_i: (n_i, 1)
    z_i: (n_i, 1)
    """

    P_y_i = F.softmax(y_i, dim=0)
    P_z_i = F.softmax(z_i, dim=0)
    return - torch.sum(P_y_i * torch.log(P_z_i))