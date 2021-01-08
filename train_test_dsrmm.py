import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from dsrmm_model import DSRMM
from data_reader import DataAndQuery
import os
import numpy as np
import torch.nn.functional as F
import pandas as pd
import subprocess
from sklearn.model_selection import KFold
import random
import argparse
from utils import *

parser = argparse.ArgumentParser(description='DSRMM', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#define inputs parameters
parser.add_argument('--max_query_len', type=int, default=6)
parser.add_argument('--max_meta_len', type=int, default=50)
parser.add_argument('--max_attributes_len', type=int, default=30)
parser.add_argument('--max_values_len', type=int, default=20)

#define model parameters
parser.add_argument('--emsize', type=int, default=300)
parser.add_argument('--k1', type=int, default=20)
parser.add_argument('--k2', type=int, default=20)
parser.add_argument('--k3', type=int, default=20)
parser.add_argument('--k4', type=int, default=200)
parser.add_argument('--sem_feature', type=int, default=100)
parser.add_argument('--STR', type=bool, default=False)
parser.add_argument('--nbins', type=int, default=5)

#define training and testing parameters
parser.add_argument('--device', type=int, default=0) #set to -1 for cpu
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--data_path', default='wikiTables')
parser.add_argument('--inter_folder', default='inter_folder')
parser.add_argument('--values_file', default='data_fields_with_struct_values2.json')
parser.add_argument('--use_max_ndcg', type=bool, default=True)
parser.add_argument('--lr', type=float, default=0.001)

parser.add_argument('--word_embedding', default='/home/mohamed/PycharmProjects/glove.6B/glove.6B.300d.txt')

args = parser.parse_args()
print(torch.cuda.current_device())
torch.cuda.set_device(args.device)
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(args.device))
print(torch.cuda.is_available())
if args.device>-1:
    args.device='cuda:'+str(args.device)
else:
    args.device='cpu'

out_str = str(args)
print(out_str)

if not os.path.exists(args.inter_folder):
    os.mkdir(args.inter_folder)


args.mu = kernal_mus(args.nbins)
args.sigma = kernel_sigmas(args.nbins)


text_file = open(os.path.join(args.data_path,"qrels.txt"), "r")
lines = text_file.readlines()

queries_id_qrels = []
list_lines_qrels = []

for line in lines:
    # print(line)
    line = line[0:len(line) - 1]
    aa = line.split('\t')
    queries_id_qrels += [aa[0]]
    list_lines_qrels.append(aa)

def test_output(test_iter, model):

    #model = load_checkpoint_for_eval(model, save_path)
    # move the model to GPU if has one
    model=model.to(args.device)

    # need this for dropout
    model.eval()

    all_values_struct = test_iter.dataset.all_values_struct

    epoch_loss = 0
    num_batches = len(test_iter)
    all_outputs = []
    all_labels=[]
    i = 0
    for batch_desc,batch_att,batch_query, labels,batch_semantic,batch_values in test_iter:
        batch_desc, batch_att, batch_query, labels,batch_semantic,batch_values = batch_desc.to(args.device), batch_att.to(args.device), batch_query.to(args.device), labels.to(args.device), batch_semantic.to(args.device), batch_values.to(args.device)
        batch_query = torch.squeeze(batch_query)
        batch_desc = torch.squeeze(batch_desc)
        batch_att = torch.squeeze(batch_att)
        batch_values = torch.squeeze(batch_values)

        batch_desc = torch.cat([batch_desc, batch_att,batch_values], 1)

        if (i+1)*batch_size<=len(all_values_struct):
            indices=np.arange(i*batch_size,(i+1)*batch_size)
        else:
            indices = np.arange(i * batch_size, len(all_values_struct))
        batch_values_struct = [all_values_struct[ii] for ii in indices]

        outputs = model(batch_query, batch_desc,batch_values_struct,batch_semantic).to(args.device)
        # print(outputs)
        # labels=torch.FloatTensor(labels)
        #labels = labels / 2

        loss = loss_function(outputs, labels.float())
        #loss = listnet_loss(labels.float(), outputs)
        epoch_loss += loss.item()

        all_outputs += outputs.tolist()
        all_labels += labels.tolist()

        i+=1

    losslogger = epoch_loss / num_batches

    #print(f'Testing loss = {losslogger}')

    return all_outputs,losslogger,all_labels

loss_function=nn.MSELoss()


batch_size=50

kfold = KFold(5, True, None)
data=read_file_for_nfcg(os.path.join(args.data_path,"all.txt"))

NUM_EPOCH=args.epochs

start_epoch=0

final_results=[]
final_results_map=[]
final_results_mrr=[]

for _ in range(1):

    all_test_max_ndcg = []
    all_test_max_map = []
    all_test_max_mrr = []

    split_id = 0

    for train, test in kfold.split(data):
        ndcg_train = []
        ndcg_test = []

        split_id+=1

        output_qrels_train = os.path.join(args.inter_folder, 'qrels_train' + str(split_id) + '.txt')
        qrel_for_data(data[train], list_lines_qrels, output_qrels_train)
        output_qrels_test = os.path.join(args.inter_folder, 'qrels_test' + str(split_id) + '.txt')
        qrel_for_data(data[test], list_lines_qrels, output_qrels_test)

        train_file_name = os.path.join(args.inter_folder, 'train1_' + str(split_id) + '.txt')
        np.savetxt(train_file_name, data[train], fmt="%s", delimiter='\t')
        test_file_name = os.path.join(args.inter_folder, 'test1_' + str(split_id) + '.txt')
        np.savetxt(test_file_name, data[test], fmt="%s", delimiter='\t')

        output_train_ndcg = os.path.join(args.inter_folder, 'train1_ndcg_' + str(split_id) + '.txt')
        output_test_ndcg = os.path.join(args.inter_folder, 'test1_ndcg_' + str(split_id) + '.txt')

        train_dataset = DataAndQuery(train_file_name,True,None,None,None,output_train_ndcg,args)
        print(len(train_dataset))
        train_iter = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)

        args.index_to_word=train_dataset.index_to_word
        args.wv=train_dataset.wv

        model = DSRMM(args).to(args.device)

        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-8)
        #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-8)
        #losslogger=np.inf
        #save_path = './model2.pt'

        #model, optimizer, start_epoch, losslogger=load_checkpoint(model, optimizer, losslogger, save_path)

        test_dataset = DataAndQuery(test_file_name, False, train_dataset.wv, train_dataset.word_to_index,
                                    train_dataset.index_to_word, output_test_ndcg, args)
        print(len(test_dataset))
        test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        inter_train = read_file_for_nfcg(output_train_ndcg)
        inter_test = read_file_for_nfcg(output_test_ndcg)

        all_att = train_dataset.all_att
        all_desc = train_dataset.all_desc
        all_query = train_dataset.all_query
        all_query_labels = train_dataset.all_query_labels
        all_labels = np.array(train_dataset.labels)
        all_semantic = train_dataset.all_semantic
        all_values = train_dataset.all_values
        all_values_struct = train_dataset.all_values_struct


        dict_label_pos = {}

        for l in range(1, 61):
            dict_label_pos[l] = [i for i, num in enumerate(all_query_labels) if num == l]

        loss_train=[]
        loss_test=[]
        max_test_ndcg=0
        max_test_map = 0
        max_test_mrr = 0

        for epoch in range(start_epoch, NUM_EPOCH + start_epoch):
            model.train()
            epoch_loss = 0
            # num_batches = len(train_iter)
            num_batches = 60
            all_outputs = []

            for l in range(1, 61):

                if len(dict_label_pos[l]) > 0:
                    batch_desc = all_desc[dict_label_pos[l]]
                    batch_att = all_att[dict_label_pos[l]]
                    batch_query = all_query[dict_label_pos[l]]
                    batch_semantic = all_semantic[dict_label_pos[l]]
                    batch_values = all_values[dict_label_pos[l]]
                    batch_values_struct = [all_values_struct[ii] for ii in dict_label_pos[l]]

                    batch_desc = batch_desc.to(args.device)
                    batch_att = batch_att.to(args.device)
                    batch_query = batch_query.to(args.device)
                    batch_semantic = batch_semantic.to(args.device)
                    batch_values = batch_values.to(args.device)

                    labels = torch.tensor(all_labels[dict_label_pos[l]])
                    labels = labels.to(args.device)

                    batch_query=torch.squeeze(batch_query)
                    batch_desc = torch.squeeze(batch_desc)
                    batch_att = torch.squeeze(batch_att)
                    batch_values = torch.squeeze(batch_values)

                    batch_desc=torch.cat([batch_desc,batch_att,batch_values],1)

                    outputs = model(batch_query,batch_desc,batch_values_struct,batch_semantic).to(args.device)
                    # labels=torch.FloatTensor(labels)
                    all_outputs += outputs.tolist()

                    #loss = loss_function(outputs, labels.float()).to(args.device)
                    loss = listnet_loss(labels.float(), outputs).to(args.device)
                    epoch_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


            losslogger = epoch_loss / num_batches

            #print(f'Training epoch = {epoch + 1}, epoch loss = {losslogger}')
            ranking_scores_file = os.path.join(args.inter_folder, 'scores.txt')
            train_ndcg = calculate_ndcg(inter_train, ranking_scores_file, all_outputs, output_qrels_train)
            ndcg_train.append(train_ndcg)

            outputs_test, testing_loss, _ = test_output(test_iter, model)
            test_ndcg,test_map,test_mrr = calculate_metrics(inter_test, ranking_scores_file, outputs_test, output_qrels_test)

            if test_ndcg>max_test_ndcg:
                max_test_ndcg=test_ndcg
                max_test_map = test_map
                max_test_mrr = test_mrr

            ndcg_test.append(test_ndcg)
            print(test_ndcg)

            loss_train.append(losslogger)
            loss_test.append(testing_loss)

        if args.use_max_ndcg:
            all_test_max_ndcg.append(max_test_ndcg)
            all_test_max_map.append(max_test_map)
            all_test_max_mrr.append(max_test_mrr)
        else:

            all_test_max_ndcg.append(test_ndcg)
            all_test_max_map.append(test_map)
            all_test_max_mrr.append(test_mrr)

        print(all_test_max_ndcg)


    final_results+=all_test_max_ndcg
    final_results_map += all_test_max_map
    final_results_mrr += all_test_max_mrr

print('final results \n')

print(final_results)
print(len(final_results))
print('mean ndcg={}'.format(np.mean(final_results)))
print('std ndcg={}'.format(np.std(final_results)))

print(final_results_map)
print(len(final_results_map))
print('mean map={}'.format(np.mean(final_results_map)))
print('std map={}'.format(np.std(final_results_map)))

print(final_results_mrr)
print(len(final_results_mrr))
print('mean mrr={}'.format(np.mean(final_results_mrr)))
print('std mrr={}'.format(np.std(final_results_mrr)))





