import torch
import torch.nn as nn


import numpy as np
import torch.nn.functional as f
from torch.autograd import Variable
import torch.nn.functional as F


class DSRMM(nn.Module):

    def __init__(self, args):
        """"Constructor of the class."""
        super(DSRMM, self).__init__()

        self.wv=args.wv
        self.index_to_word=args.index_to_word

        self.input_dim=args.emsize
        self.device=args.device

        self.STR=args.STR

        self.nbins = args.nbins
        #self.bins = [-1.0, -0.5, 0, 0.5, 1.0, 1.0]
        self.bins = [-0.75, -0.25, 0.25, 0.75, 1.0, 1.0]

        self.gating_network = GatingNetwork(args.emsize)


        self.conv1 = nn.Conv2d(self.input_dim, args.k1, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(self.input_dim, args.k2, (3, 5), padding=(1, 2))
        self.conv3 = nn.Conv2d(self.input_dim, args.k3, (3, 7), padding=(1, 3))
        self.relu = nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv_all = nn.Conv2d(args.k1+args.k2+args.k3, args.k4, (3, 3), padding=1)
        self.conv_dim = nn.Conv2d(args.k4, args.sem_feature, (1, 1))

        self.conv_uni = nn.Sequential(
            nn.Conv2d(1, args.emsize, (1, self.input_dim)),
            nn.ReLU()
        )

        tensor_mu = torch.FloatTensor(args.mu).to(self.device)
        tensor_sigma = torch.FloatTensor(args.sigma).to(self.device)

        self.mu = Variable(tensor_mu, requires_grad=False).view(1, 1, 1, self.nbins)
        self.sigma = Variable(tensor_sigma, requires_grad=False).view(1, 1, 1, self.nbins)

        if args.STR:
            self.output3 = nn.Linear(args.sem_feature+args.nbins*args.max_query_len+39, 1,True)
        else:
            self.output3 = nn.Linear(args.sem_feature+args.nbins*args.max_query_len, 1,True)

    def get_intersect_matrix(self, q_embed, d_embed):
        sim = torch.bmm(q_embed, d_embed).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[2], 1)
        pooling_value = torch.exp((- ((sim - self.mu) ** 2) / (self.sigma ** 2) / 2))
        pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * 0.01
        #log_pooling_sum = torch.sum(log_pooling_sum, 1)
        return log_pooling_sum

    def get_intersect_matrix_with_cos(self, q_embed, d_embed):


        sim = f.cosine_similarity(q_embed, d_embed, 3).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[2], 1)
        pooling_value = torch.exp((- ((sim - self.mu) ** 2) / (self.sigma ** 2) / 2))
        pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * 0.01
        #log_pooling_sum = torch.sum(log_pooling_sum, 1)
        return log_pooling_sum


    def to_embedding(self,input):
        shape_input = list(input.shape)

        em = input.view(-1)
        list_of_embeddings = []
        for key in em:
            list_of_embeddings += self.wv[self.index_to_word[key]].tolist()
        list_of_embeddings = torch.Tensor(list_of_embeddings)
        embeds = list_of_embeddings.view(shape_input[0], shape_input[1],
                                         self.input_dim).to(self.device)

        return embeds


    def forward(self, batch_queries, batch_docs,batch_values_struct,batch_semantic):



        num_docs, dlen = batch_docs.shape[0], batch_docs.shape[1]


        emb_query = self.to_embedding(batch_queries)
        emb_desc = self.to_embedding(batch_docs)

        all_tables = []

        for sample in batch_values_struct:
            l_table = []
            for instance in sample:
                instance = torch.Tensor(instance).unsqueeze(0)
                instance = instance.type(torch.int64)
                emb_instance = self.to_embedding(instance)
                emb_instance = torch.mean(emb_instance, dim=1)
                l_table.append(emb_instance.tolist())
            all_tables.append(l_table)

        all_tables = torch.Tensor(all_tables).squeeze(2).to(self.device)
        emb_desc=torch.cat([emb_desc,all_tables],dim=1)

        desc_att_shape = emb_desc.shape
        query_shape = emb_query.shape

        embedded_docs = torch.stack([emb_desc] * query_shape[1], dim=1).to(self.device)
        embedded_queries = torch.stack([emb_query] * desc_att_shape[1], dim=2).to(self.device)

        qwu_embed = torch.transpose(
            torch.squeeze(self.conv_uni(emb_query.view(emb_query.size()[0], 1, -1, self.input_dim))), 1,
            2) + 0.000000001

        dwu_embed = torch.squeeze(
            self.conv_uni(emb_desc.view(emb_desc.size()[0], 1, -1, self.input_dim))) + 0.000000001


        qwu_embed_norm = F.normalize(qwu_embed, p=2, dim=2, eps=1e-10)
        dwu_embed_norm = F.normalize(dwu_embed, p=2, dim=1, eps=1e-10)

        #log_pooling_sum_wwuu = self.get_intersect_matrix(qwu_embed_norm, dwu_embed_norm)

        dwu_embed_norm = torch.stack([dwu_embed_norm] * query_shape[1], dim=1).to(self.device)
        dwu_embed_norm = dwu_embed_norm.permute(0, 1, 3, 2)
        qwu_embed_norm = torch.stack([qwu_embed_norm] * desc_att_shape[1], dim=2).to(self.device)

        log_pooling_sum_wwuu = self.get_intersect_matrix_with_cos(qwu_embed_norm, dwu_embed_norm)

        term_weights = self.gating_network(emb_query)
        term_weights = torch.stack([term_weights] * self.nbins, dim=2).to(self.device)
        hist_feat=term_weights*log_pooling_sum_wwuu
        hist_feat=hist_feat.view([num_docs,-1])

        new_input = embedded_docs * embedded_queries

        new_input = new_input.permute(0, 3, 1, 2)
        convoluted_feat1 = self.conv1(new_input)
        convoluted_feat2 = self.conv2(new_input)
        convoluted_feat3 = self.conv3(new_input)
        convoluted_feat = self.relu(torch.cat((convoluted_feat1, convoluted_feat2, convoluted_feat3), 1))

        pooled_feat = self.pool(convoluted_feat)
        conv_all_feat = self.conv_all(pooled_feat)
        conv_all_feat = self.relu(conv_all_feat)

        conv_all_feat = self.conv_dim(conv_all_feat)

        conv_all_feat = conv_all_feat.permute(0, 2, 3, 1)

        max_pooled_feat = torch.max(conv_all_feat, 2)[0]
        max_pooled_feat = torch.max(max_pooled_feat, 1)[0]

        semantic_input = batch_semantic.type(torch.float32)
        if self.STR:
            final_feat = torch.cat((max_pooled_feat,hist_feat, semantic_input), dim=1)

        else:
            final_feat = torch.cat((max_pooled_feat, hist_feat), dim=1)

        final_score = self.output3(final_feat).squeeze(-1)


        return final_score


class GatingNetwork(nn.Module):
    """Term gating network"""

    def __init__(self, emsize):
        """"Constructor of the class"""
        super(GatingNetwork, self).__init__()
        self.weight = nn.Linear(emsize, 1)

    def forward(self, term_embeddings):
        """"Defines the forward computation of the gating network layer."""
        dot_out = self.weight(term_embeddings).squeeze(2)
        return f.softmax(dot_out, 1)
