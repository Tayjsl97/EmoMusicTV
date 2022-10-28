"""
Created on Fri Oct 29 16:28:18 2022
@author: Shulei Ji
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.size=size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderFirstBlock1(nn.Module):
    def __init__(self, size, self_attn, dropout):
        super(DecoderFirstBlock1, self).__init__()
        self.self_attn = self_attn
        self.sublayer = clones(SublayerConnection(size, dropout), 1)

    def forward(self, x, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        return x


class DecoderLastBlock3(nn.Module):
    def __init__(self, size, feed_forward, dropout):
        super(DecoderLastBlock3, self).__init__()
        self.size = size
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 1)

    def forward(self, x):
        return self.sublayer[0](x, self.feed_forward)


class DecoderLayer2(nn.Module):
    def __init__(self, size, src_attn, dropout):
        super(DecoderLayer2, self).__init__()
        self.src_attn = src_attn
        self.sublayer = clones(SublayerConnection(size, dropout), 1)

    def forward(self, x, memory, src_mask):
        m = memory
        return self.sublayer[0](x, lambda x: self.src_attn(x, m, m, src_mask))


class DecoderLayer31(nn.Module):
    def __init__(self, size, feed_forward, self_attn, dropout):
        super(DecoderLayer31, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, tgt_mask):
        x = self.sublayer[0](x, self.feed_forward)
        return self.sublayer[1](x, lambda x: self.self_attn(x, x, x, tgt_mask))


def full_mask(batch, seqLen1, seqLen2):
    attn_shape = (batch, seqLen1, seqLen2)
    encoder_mask = np.ones(attn_shape)
    return (torch.from_numpy(encoder_mask)).to(device)


def triple_mask(batch,l_decoder_seq):
    decoder_attn_shape = (batch, l_decoder_seq, l_decoder_seq)
    decoder_mask = np.triu(np.ones(decoder_attn_shape), k=1).astype('uint8')
    return (torch.from_numpy(decoder_mask) == 0).to(device)


class MusicTV(nn.Module):
    def __init__(self,N,h,m_size,c_size,d_ff,hidden_size,latent_size,dropout):
        super(MusicTV,self).__init__()
        c=copy.deepcopy
        self.N=N
        self.m_size = m_size
        self.c_size = c_size
        self.latent_size=latent_size
        self.decoder_input=128
        self.melody_input = nn.Embedding(107,m_size)
        self.melody_linear=nn.Linear(m_size,48)
        self.melody_embedd = nn.Sequential(c(PositionalEncoding(c_size, dropout)))
        self.chord_embedd = nn.Sequential(c(PositionalEncoding(c_size, dropout)))
        self.chord_attn = MultiHeadedAttention(h, c_size)
        self.melody_attn = MultiHeadedAttention(h, c_size)
        self.chord_feed_forward = PositionwiseFeedForward(c_size, d_ff, dropout)
        self.melody_feed_forward = PositionwiseFeedForward(c_size, d_ff, dropout)
        self.chord_encoder_layer = EncoderLayer(c_size, c(self.chord_attn), c(self.chord_feed_forward), dropout)
        self.chord_encoder_layers = clones(self.chord_encoder_layer, N)
        self.melody_encoder_layer = EncoderLayer(c_size, c(self.melody_attn), c(self.melody_feed_forward), dropout)
        self.melody_encoder_layers = clones(self.melody_encoder_layer, N)
        self.melody_encoder_norm = LayerNorm(self.melody_encoder_layer.size)
        self.chord_encoder_norm = LayerNorm(self.chord_encoder_layer.size)
        self.pValence1 = nn.Linear(5,hidden_size)
        self.pValence2 = nn.Linear(hidden_size, c_size)
        self.prior_linear=nn.Linear(c_size*2,c_size)
        self.recog_linear = nn.Linear(c_size * 2, c_size)
        self.sentiment_lstm = nn.LSTM(5, hidden_size, 2, batch_first=True, bidirectional=True)
        self.P_hidden2mean1_m = nn.Linear(c_size * 3, latent_size)
        self.P_hidden2logv1_m = nn.Linear(c_size * 3, latent_size)
        self.P_hidden2mean1_c = nn.Linear(c_size * 3, latent_size)
        self.P_hidden2logv1_c = nn.Linear(c_size * 3, latent_size)
        self.P_hidden2mean2_m = nn.Linear(c_size, latent_size)
        self.P_hidden2logv2_m = nn.Linear(c_size, latent_size)
        self.P_hidden2mean2_c = nn.Linear(c_size, latent_size)
        self.P_hidden2logv2_c = nn.Linear(c_size, latent_size)
        self.B_hidden2mean1_m = nn.Linear(self.decoder_input + hidden_size + c_size + 5 + 8, latent_size)
        self.B_hidden2logv1_m = nn.Linear(self.decoder_input + hidden_size + c_size + 5 + 8, latent_size)
        self.B_hidden2mean1_c = nn.Linear(self.decoder_input + hidden_size + c_size + 5 + 8, latent_size)
        self.B_hidden2logv1_c = nn.Linear(self.decoder_input + hidden_size + c_size + 5 + 8, latent_size)
        self.B_hidden2mean2_m = nn.Linear(self.decoder_input + 5 + 8, latent_size)
        self.B_hidden2logv2_m = nn.Linear(self.decoder_input + 5 + 8, latent_size)
        self.B_hidden2mean2_c = nn.Linear(self.decoder_input + 5 + 8, latent_size)
        self.B_hidden2logv2_c = nn.Linear(self.decoder_input + 5 + 8, latent_size)
        self.memory_dim_add = nn.Linear(self.c_size, self.decoder_input)
        self.decoder_input_dim_reduct = nn.Linear(self.latent_size * 2 + c_size + 5 + 8, self.decoder_input)
        ### melody decoder ###
        self.m_decoder_attn = MultiHeadedAttention(h, self.decoder_input)
        self.m_decoder_feed_forward = PositionwiseFeedForward(self.decoder_input, d_ff, dropout)
        self.m_decoder_layer1 = DecoderFirstBlock1(self.decoder_input, c(self.m_decoder_attn),dropout)
        self.m_decoder_layer2_melody = DecoderLayer2(self.decoder_input, c(self.m_decoder_attn), dropout)
        self.m_decoder_layer2_chord = DecoderLayer2(self.decoder_input, c(self.m_decoder_attn), dropout)
        self.m_decoder_layer31 = DecoderLayer31(self.decoder_input, c(self.m_decoder_feed_forward),c(self.m_decoder_attn), dropout)
        self.m_decoder_layer4 = DecoderLastBlock3(self.decoder_input, c(self.m_decoder_feed_forward), dropout)
        self.m_decoder_layers_2_melody = clones(self.m_decoder_layer2_melody, N)
        self.m_decoder_layers_2_chord = clones(self.m_decoder_layer2_chord, N)
        self.m_decoder_layers_31 = clones(self.m_decoder_layer31, N-1)
        self.m_cross_attention_linear=nn.Linear(self.decoder_input*2,self.decoder_input)
        self.m_decoder_norm = LayerNorm(self.decoder_input)
        ### chord decoder ###
        self.c_decoder_attn = MultiHeadedAttention(h, self.decoder_input)
        self.c_decoder_feed_forward = PositionwiseFeedForward(self.decoder_input, d_ff, dropout)
        self.c_decoder_layer1 = DecoderFirstBlock1(self.decoder_input, c(self.c_decoder_attn), dropout)
        self.c_decoder_layer2_melody = DecoderLayer2(self.decoder_input, c(self.c_decoder_attn), dropout)
        self.c_decoder_layer2_chord = DecoderLayer2(self.decoder_input, c(self.c_decoder_attn), dropout)
        self.c_decoder_layer31 = DecoderLayer31(self.decoder_input, c(self.c_decoder_feed_forward), c(self.c_decoder_attn), dropout)
        self.c_decoder_layer4 = DecoderLastBlock3(self.decoder_input, c(self.c_decoder_feed_forward), dropout)
        self.c_decoder_layers_2_melody = clones(self.c_decoder_layer2_melody, N)
        self.c_decoder_layers_2_chord = clones(self.c_decoder_layer2_chord, N)
        self.c_decoder_layers_31 = clones(self.c_decoder_layer31, N - 1)
        self.c_cross_attention_linear = nn.Linear(self.decoder_input * 2, self.decoder_input)
        self.c_decoder_norm = LayerNorm(self.decoder_input)
        ### predict ###
        self.melody_note=nn.Linear(self.decoder_input,107)
        self.chord_type = nn.Linear(self.decoder_input, 7)
        self.chord_root = nn.Linear(self.decoder_input, 41)
        self.softmax=nn.LogSoftmax(dim=-1)

    def chord_encoder(self,src,src_mask):
        src = self.chord_embedd(src)
        for layer in self.chord_encoder_layers:
            src=layer(src,src_mask)
        src=self.chord_encoder_norm(src)
        return src,src.mean(dim=1)

    def melody_encoder(self,src,src_mask):
        src = self.melody_embedd(self.melody_linear(self.melody_input(src)))
        for layer in self.melody_encoder_layers:
            src=layer(src,src_mask)
        src=self.melody_encoder_norm(src)
        return src,src.mean(dim=1)

    def pValence_encoder(self,s_p):
        return self.pValence2(self.pValence1(s_p.squeeze(1)))

    def getGT(self,melody,chord):
        GT_melody=torch.zeros([melody.shape[0],1]).long().to(device)
        GT_type = torch.zeros([melody.shape[0],1]).long().to(device)
        GT_root = torch.zeros([melody.shape[0],1]).long().to(device)
        bar_index=[]
        melody_one=melody[0].cpu().tolist()
        chord_cnt=1
        for i in range(len(melody_one)):
            if melody_one[i]>=99:
                continue
            elif melody_one[i]==0 and i!=len(melody_one)-1:
                bar_index.append(GT_melody.shape[1])
                GT_bar=torch.zeros([melody.shape[0],1]).long().to(device)
                _, GT_type_i = chord[:,chord_cnt:chord_cnt+2,:].narrow(-1, 0, 7).topk(1)
                _, GT_root_i = chord[:,chord_cnt:chord_cnt+2,:].narrow(-1, 7, 41).topk(1)
                GT_type_i = GT_type_i.squeeze(-1)
                GT_root_i = GT_root_i.squeeze(-1)
                GT_melody=torch.cat((GT_melody, GT_bar),dim=1)
                GT_type = torch.cat((GT_type, GT_type_i),dim=1)
                GT_root = torch.cat((GT_root, GT_root_i), dim=1)
                chord_cnt+=2
            else:
                GT_melody = torch.cat((GT_melody,melody.narrow(1,i,1)),dim=1)
        return GT_melody[:,2:],GT_type[:,1:],GT_root[:,1:],bar_index

    def P_recognitionNet(self,melody_hidden,chord_hidden,s_p_hidden):
        concat = torch.cat((melody_hidden,chord_hidden,s_p_hidden), -1)
        m_mean = self.P_hidden2mean1_m(concat)
        m_logv = self.P_hidden2logv1_m(concat)
        c_mean = self.P_hidden2mean1_c(concat)
        c_logv = self.P_hidden2logv1_c(concat)
        return m_mean, m_logv,c_mean,c_logv

    def P_priorNet(self,melody_hidden,chord_hidden,s_p_hidden):
        temp=self.prior_linear(torch.cat((melody_hidden,chord_hidden),dim=-1))
        temp=self.prior_linear(torch.cat((temp,s_p_hidden),dim=-1))
        m_mean = self.P_hidden2mean2_m(temp)
        m_logv = self.P_hidden2logv2_m(temp)
        c_mean = self.P_hidden2mean2_c(temp)
        c_logv = self.P_hidden2logv2_c(temp)
        return m_mean, m_logv,c_mean,c_logv

    def B_recognitionNet(self,history_m,history_c,s_b,r_s,r_m,r_c,timeSign):
        m_mean=0; m_logv=0; c_mean=0; c_logv=0
        if torch.is_tensor(history_m) or history_m is None:
            if history_m is None:
                history_m=torch.zeros([s_b.shape[0],self.decoder_input]).to(device)
            concat = torch.cat((history_m, s_b, r_s, r_m, timeSign), -1)  # 64*1029  128+525
            m_mean = self.B_hidden2mean1_m(concat)
            m_logv = self.B_hidden2logv1_m(concat)
        if torch.is_tensor(history_c) or history_c is None:
            if history_c is None:
                history_c=torch.zeros([s_b.shape[0],self.decoder_input]).to(device)
            concat = torch.cat((history_c, s_b, r_s, r_c, timeSign), -1)  # 64*1029  128+525
            c_mean = self.B_hidden2mean1_c(concat)
            c_logv = self.B_hidden2logv1_c(concat)
        return m_mean,m_logv,c_mean,c_logv

    def B_priorNet(self,history_m,history_c,s_b,timeSign):
        m_mean = 0;m_logv = 0;c_mean = 0;c_logv = 0
        if torch.is_tensor(history_m) or history_m is None:
            if history_m is None:
                history_m = torch.zeros([s_b.shape[0], self.decoder_input]).to(device)
            concat = torch.cat((history_m, s_b, timeSign), -1)
            m_mean = self.B_hidden2mean2_m(concat)
            m_logv = self.B_hidden2logv2_m(concat)
        if torch.is_tensor(history_c) or history_c is None:
            if history_c is None:
                history_c = torch.zeros([s_b.shape[0], self.decoder_input]).to(device)
            concat = torch.cat((history_c, s_b, timeSign), -1)
            c_mean = self.B_hidden2mean2_c(concat)
            c_logv = self.B_hidden2logv2_c(concat)
        return m_mean, m_logv, c_mean, c_logv

    def reparameterize(self,mean,logv):
        std = torch.exp(0.5 * logv)
        z = torch.randn([mean.shape[0], self.latent_size]).to(device)
        z = z * std + mean  # 64*64
        return z

    def reverse_sentiment(self,x,i):
        reversed_s=torch.flip(x[:,i:,:],dims=[1])
        output,hidden=self.sentiment_lstm(reversed_s,None)
        hidden=hidden[0].mean(dim=0)
        return hidden

    def reverse_melody(self,melody,bar_cnt,bar_index):
        melody = melody[:, bar_index[bar_cnt]:]
        melody_mask = full_mask(melody.shape[0], melody.shape[1], melody.shape[1])
        _, m_hidden = self.melody_encoder(melody, melody_mask)
        return m_hidden

    def reverse_chord(self,chord,bar_cnt):
        chord=chord[:,bar_cnt*2:,:]
        chord_mask=full_mask(chord.shape[0],chord.shape[1],chord.shape[1])
        _, c_hidden = self.chord_encoder(chord, chord_mask)
        return c_hidden

    def decoder(self, melody, chord, melody_memory, chord_memory, bar_index, z_p_m,z_p_c, S_B):
        x_embed=self.melody_embedd(self.melody_linear(self.melody_input(melody)))
        chord_embed=self.chord_embedd(chord)
        history_m=None;history_c=None
        z_list_m=[];z_list_c=[]
        time_cnt=0
        for i in range(melody.shape[1]):
            if melody[0,i]>=99:
                time_cnt+=1
        # print("mLen: ",melody.shape[1]-time_cnt-1)
        new_melody = torch.zeros([melody.shape[0], melody.shape[1]-time_cnt-1, self.decoder_input]).to(device)
        new_chord = torch.zeros([melody.shape[0], chord.shape[1]-1, self.decoder_input]).to(device)
        melody_mask = triple_mask(new_melody.shape[0], new_melody.shape[1])
        chord_mask = triple_mask(new_chord.shape[0],new_chord.shape[1])
        bar_cnt=0;time_cnt=0
        for i in range(melody.shape[1]-1):
            if melody[0,i]>=99:
                timeSign=torch.nn.functional.one_hot(melody[:,i]-99,num_classes=8).float().to(device)
                time_cnt+=1
            if melody[0,i]==0:
                r_s=self.reverse_sentiment(S_B,bar_cnt)
                r_c=self.reverse_chord(chord,bar_cnt)
                r_m=self.reverse_melody(melody,bar_cnt,bar_index)
                m_b_mean,m_b_logv,c_b_mean,c_b_logv=self.B_recognitionNet(history_m,history_c, S_B[:, bar_cnt, :], r_s,r_m,r_c, timeSign)
                z_b_m = self.reparameterize(m_b_mean,m_b_logv)
                z_b_c = self.reparameterize(c_b_mean, c_b_logv)
                m_b_mean_p,m_b_logv_p,c_b_mean_p,c_b_logv_p=self.B_priorNet(history_m,history_c, S_B[:,bar_cnt,:],timeSign)
                z_list_m.append([m_b_mean,m_b_logv,m_b_mean_p,m_b_logv_p])
                z_list_c.append([c_b_mean,c_b_logv,c_b_mean_p,c_b_logv_p])
                new_chord[:, bar_cnt*2, :] = self.decoder_input_dim_reduct(
                    torch.cat((chord_embed[:, 2 * bar_cnt, :], z_b_c, z_p_c, S_B[:, bar_cnt, :], timeSign), -1))
                new_chord[:, bar_cnt*2 + 1, :] = self.decoder_input_dim_reduct(
                    torch.cat((chord_embed[:, 2 * bar_cnt + 1, :], z_b_c, z_p_c, S_B[:, bar_cnt, :], timeSign), -1))
                j = i
                while True:
                    new_melody[:, j - time_cnt, :] = self.decoder_input_dim_reduct(
                        torch.cat((x_embed[:,j, :], z_b_m, z_p_m, S_B[:, bar_cnt, :], timeSign), -1))
                    j+=1
                    if j==melody.shape[1]-1 or melody[0,j]==0 or melody[0,j]>=99:
                        break
                chord_temp_x=self.c_decoder_layer1(new_chord,chord_mask)
                melody_temp_x = self.m_decoder_layer1(new_melody, melody_mask)
                history_c = chord_temp_x[:,bar_cnt*2 + 1,:].detach()
                history_m = melody_temp_x[:,j-1-time_cnt, :].detach()
                bar_cnt += 1
                # print("history: ",history.shape)
        chord_x=chord_temp_x
        for i in range(self.N):
            x_melody=self.c_decoder_layers_2_melody[i](chord_x,melody_memory,full_mask(chord_x.shape[0],chord_x.shape[1],melody_memory.shape[1]))
            x_chord=self.c_decoder_layers_2_chord[i](chord_x,chord_memory,full_mask(chord_x.shape[0],chord_x.shape[1],chord_memory.shape[1]))
            chord_x=self.c_cross_attention_linear(torch.cat((x_melody,x_chord),dim=-1))
            if i<self.N-1:
                chord_x=self.c_decoder_layers_31[i](chord_x,chord_mask)
        chord_x=self.c_decoder_layer4(chord_x)
        chord_x=self.c_decoder_norm(chord_x)
        melody_x = melody_temp_x
        for i in range(self.N):
            x_melody = self.m_decoder_layers_2_melody[i](melody_x, melody_memory,full_mask(melody_x.shape[0], melody_x.shape[1], melody_memory.shape[1]))
            x_chord = self.m_decoder_layers_2_chord[i](melody_x, chord_memory,full_mask(melody_x.shape[0], melody_x.shape[1], chord_memory.shape[1]))
            melody_x = self.m_cross_attention_linear(torch.cat((x_melody, x_chord), dim=-1))
            if i < self.N - 1:
                melody_x = self.m_decoder_layers_31[i](melody_x, melody_mask)
        melody_x = self.m_decoder_layer4(melody_x)
        melody_x = self.m_decoder_norm(melody_x)
        return melody_x,chord_x,z_list_m,z_list_c

    def forward(self,melody_pre,melody_post,chord_pre,chord_post,s_p,S_B):
        GT_note,GT_type,GT_root,bar_index = self.getGT(melody_post,chord_post)
        melody_memory,prior_melody_hidden=self.melody_encoder(melody_pre,full_mask(melody_pre.shape[0],melody_pre.shape[1],melody_pre.shape[1]))
        chord_memory,prior_chord_hidden=self.chord_encoder(chord_pre,full_mask(chord_pre.shape[0],chord_pre.shape[1],chord_pre.shape[1]))
        _, post_melody_hidden = self.melody_encoder(melody_post, full_mask(melody_post.shape[0], melody_post.shape[1],melody_post.shape[1]))
        _, post_chord_hidden = self.chord_encoder(chord_post, full_mask(chord_post.shape[0], chord_post.shape[1],chord_post.shape[1]))
        recog_melody_hidden=self.recog_linear(torch.cat((prior_melody_hidden,post_melody_hidden),dim=-1))
        recog_chord_hidden=self.recog_linear(torch.cat((prior_chord_hidden,post_chord_hidden),dim=-1))
        melody_memory=self.memory_dim_add(melody_memory)
        chord_memory=self.memory_dim_add(chord_memory)
        pValence_hidden = self.pValence_encoder(s_p)
        m_mean, m_logv,c_mean,c_logv = self.P_recognitionNet(recog_melody_hidden,recog_chord_hidden,pValence_hidden)
        m_mean_p, m_logv_p,c_mean_p,c_logv_p = self.P_priorNet(prior_melody_hidden,prior_chord_hidden,pValence_hidden)
        KL_loss_m = -0.5 * torch.sum(1 + 2 * (m_logv - m_logv_p) - (
            (m_logv.exp().pow(2) + (m_mean_p - m_mean).pow(2)).div(m_logv_p.exp().pow(2))),
            1).mean()
        KL_loss_c = -0.5 * torch.sum(1 + 2 * (c_logv - c_logv_p) - (
            (c_logv.exp().pow(2) + (c_mean_p - c_mean).pow(2)).div(c_logv_p.exp().pow(2))),
                                     1).mean()
        KL_loss=KL_loss_m+KL_loss_c
        z_p_m = self.reparameterize(m_mean,m_logv)
        z_p_c = self.reparameterize(c_mean, c_logv)
        gen_melody,gen_chord,z_list_m,z_list_c= self.decoder(melody_post,chord_post,melody_memory,chord_memory,bar_index,z_p_m,z_p_c,S_B)
        b_KL_loss_m=0
        b_KL_loss_c = 0
        for i in z_list_m:
            b_KL_loss_m += -0.5 * torch.sum(
                1 + 2 * (i[1] - i[3]) - ((i[1].exp().pow(2) + (i[2] - i[0]).pow(2)).div(i[3].exp().pow(2))),
                1).mean()
        for i in z_list_c:
            b_KL_loss_c += -0.5 * torch.sum(
                1 + 2 * (i[1] - i[3]) - ((i[1].exp().pow(2) + (i[2] - i[0]).pow(2)).div(i[3].exp().pow(2))),
                1).mean()
        b_KL_loss_m=b_KL_loss_m/len(z_list_m)
        b_KL_loss_c = b_KL_loss_c / len(z_list_c)
        b_KL_loss=b_KL_loss_m+b_KL_loss_c
        note = self.softmax(self.melody_note(gen_melody)+1e-10)
        root = self.softmax(self.chord_root(gen_chord) + 1e-10)
        type = self.softmax(self.chord_type(gen_chord) + 1e-10)
        return KL_loss,b_KL_loss,note,type,root,GT_note,GT_type,GT_root

    def softmax2chord(self,type,root):
        topk,topi=type.topk(1)
        topi=topi.squeeze(-1)
        type=torch.nn.functional.one_hot(topi,num_classes=7).float()
        topk, topi = root.topk(1)
        topi = topi.squeeze(-1)
        root = torch.nn.functional.one_hot(topi, num_classes=41).float()
        chord=torch.cat((type,root),-1)
        return chord

    def generate(self,melody_pre,chord_pre,s_p,S_B,timeSign):
        melody_memory, prior_melody_hidden = self.melody_encoder(melody_pre,full_mask(melody_pre.shape[0], melody_pre.shape[1],
                                                                           melody_pre.shape[1]))
        chord_memory, prior_chord_hidden = self.chord_encoder(chord_pre,full_mask(chord_pre.shape[0], chord_pre.shape[1],
                                                                        chord_pre.shape[1]))
        melody_memory = self.memory_dim_add(melody_memory)
        chord_memory = self.memory_dim_add(chord_memory)
        pValence_hidden = self.pValence_encoder(s_p)
        m_mean_p, m_logv_p,c_mean_p,c_logv_p = self.P_priorNet(prior_melody_hidden,prior_chord_hidden,pValence_hidden)
        z_p_m = self.reparameterize(m_mean_p,m_logv_p)
        z_p_c = self.reparameterize(c_mean_p, c_logv_p)
        barNum=S_B.shape[1]
        bar_cnt=0
        generate_notes = []; generate_chords=[]
        melody_input_x=self.melody_linear(self.melody_input(torch.zeros([S_B.shape[0],1]).long().to(device)))
        melody_new_x = torch.zeros([S_B.shape[0], 1, self.decoder_input]).to(device)
        chord_input_x = torch.zeros([S_B.shape[0], 1,48]).to(device)
        chord_input_x[:,:,0]=1;chord_input_x[:,:,-1]=1
        chord_new_x = torch.zeros([S_B.shape[0], 1, self.decoder_input]).to(device)
        i=0;j=0
        while True:
            if bar_cnt == 0:
                m_b_mean_p,m_b_logv_p,c_b_mean_p,c_b_logv_p=self.B_priorNet(None,None,S_B[:, bar_cnt, :], timeSign)
            else:
                m_b_mean_p,m_b_logv_p,c_b_mean_p,c_b_logv_p=self.B_priorNet(m_history, c_history, S_B[:, bar_cnt, :],timeSign)
            z_b_m = self.reparameterize(m_b_mean_p,m_b_logv_p)
            z_b_c = self.reparameterize(c_b_mean_p, c_b_logv_p)
            ############# chord ##############
            for h in range(2):
                chord_x=self.chord_embedd(chord_input_x)
                chord_mask = triple_mask(chord_x.shape[0], chord_x.shape[1])
                chord_new_x[:, i, :] = self.decoder_input_dim_reduct(
                    torch.cat((chord_x[:, i, :], z_b_c, z_p_c, S_B[:, bar_cnt, :], timeSign), -1))
                chord_x = self.c_decoder_layer1(chord_new_x, chord_mask)
                c_history=chord_x[:,-1,:]
                for k in range(self.N):
                    x_melody = self.c_decoder_layers_2_melody[k](chord_x, melody_memory,full_mask(chord_x.shape[0], chord_x.shape[1], melody_memory.shape[1]))
                    x_chord = self.c_decoder_layers_2_chord[k](chord_x, chord_memory,full_mask(chord_x.shape[0], chord_x.shape[1], chord_memory.shape[1]))
                    chord_x = self.c_cross_attention_linear(torch.cat((x_melody, x_chord), dim=-1))
                    if k < self.N - 1:
                        chord_x = self.c_decoder_layers_31[k](chord_x, chord_mask)
                chord_x = self.c_decoder_layer4(chord_x)
                chord_x = self.c_decoder_norm(chord_x)
                root = self.softmax(self.chord_root(chord_x.narrow(1, -1, 1)) + 1e-10)
                type = self.softmax(self.chord_type(chord_x.narrow(1, -1, 1)) + 1e-10)
                chord = self.softmax2chord(type, root)
                generate_chords.append(chord.squeeze(0).squeeze(0).cpu().tolist())
                chord_input_x = torch.cat([chord_input_x, chord], dim=1)
                temp = torch.zeros([S_B.shape[0], 1, self.decoder_input]).to(device)
                chord_new_x = torch.cat([chord_new_x, temp], dim=1)
                i+=1
            ############ melody ###############
            while True:
                melody_x = self.melody_embedd(melody_input_x)
                melody_mask=triple_mask(melody_x.shape[0],melody_x.shape[1])
                melody_new_x[:, j, :] = self.decoder_input_dim_reduct(
                    torch.cat((melody_x[:, j, :], z_b_m, z_p_m, S_B[:, bar_cnt, :],timeSign), -1))
                melody_x = self.m_decoder_layer1(melody_new_x,melody_mask)
                m_history=melody_x[:,-1,:]
                for k in range(self.N):
                    x_melody = self.m_decoder_layers_2_melody[k](melody_x, melody_memory, full_mask(melody_x.shape[0], melody_x.shape[1],melody_memory.shape[1]))
                    x_chord = self.m_decoder_layers_2_chord[k](melody_x, chord_memory, full_mask(melody_x.shape[0], melody_x.shape[1],chord_memory.shape[1]))
                    melody_x = self.m_cross_attention_linear(torch.cat((x_melody, x_chord), dim=-1))
                    if k < self.N - 1:
                        melody_x = self.m_decoder_layers_31[k](melody_x, melody_mask)
                melody_x = self.m_decoder_layer4(melody_x)
                melody_x = self.m_decoder_norm(melody_x)
                note = self.softmax(self.melody_note(melody_x)+1e-10)
                topk, topi = note.topk(1)
                topi = topi[:,j,:]
                # print(topi)
                note=topi.squeeze(0).item()
                generate_notes.append(note)
                embedd_note=self.melody_linear(self.melody_input(torch.LongTensor([note]*S_B.shape[0]).unsqueeze(-1).long().to(device)))
                melody_input_x = torch.cat([melody_input_x, embedd_note], dim=1)
                temp = torch.zeros([S_B.shape[0], 1, self.decoder_input]).to(device)
                melody_new_x=torch.cat([melody_new_x,temp],dim=1)
                j+=1
                if note==0:
                    break
            bar_cnt+=1
            if bar_cnt==barNum:
                break
        return generate_notes,generate_chords
