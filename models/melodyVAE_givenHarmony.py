"""
Created on Fri Oct 29 17:28:18 2022
@author: Shulei Ji
"""

import torch
from models.EmoMusicTV import EmoMusicTV,full_mask,triple_mask,LayerNorm

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


class melodyVAE(EmoMusicTV):
    def __init__(self,N,h,m_size,c_size,d_ff,hidden_size,latent_size,dropout):
        super().__init__(N,h,m_size,c_size,d_ff,hidden_size,latent_size,dropout)

    def P_recognitionNet(self,melody_hidden,chord_hidden,s_p_hidden):
        concat = torch.cat((melody_hidden,chord_hidden,s_p_hidden), -1)
        m_mean = self.P_hidden2mean1_m(concat)
        m_logv = self.P_hidden2logv1_m(concat)
        return m_mean, m_logv

    def P_priorNet(self,chord_hidden,pValence_hidden):
        concat = self.prior_linear(torch.cat((chord_hidden, pValence_hidden), -1))
        mean = self.P_hidden2mean2_m(concat)
        logv = self.P_hidden2logv2_m(concat)
        return mean, logv

    def B_priorNet(self,history_m,s_b,timeSign):
        if history_m is None:
            history_m = torch.zeros([s_b.shape[0], self.decoder_input]).to(device)
        concat = torch.cat((history_m, s_b, timeSign), -1)
        m_mean = self.B_hidden2mean2_m(concat)
        m_logv = self.B_hidden2logv2_m(concat)
        return m_mean, m_logv

    def B_recognitionNet(self,history_m,s_b,r_s,r_m,timeSign):
        m_mean=0; m_logv=0
        if torch.is_tensor(history_m) or history_m is None:
            if history_m is None:
                history_m=torch.zeros([s_b.shape[0],self.decoder_input]).to(device)
            concat = torch.cat((history_m, s_b, r_s, r_m, timeSign), -1)
            m_mean = self.B_hidden2mean1_m(concat)
            m_logv = self.B_hidden2logv1_m(concat)
        return m_mean,m_logv

    def reverse_melody(self,melody,bar_cnt,bar_index):
        melody=melody[:,bar_index[bar_cnt]:]
        melody_mask = full_mask(melody.shape[0], melody.shape[1], melody.shape[1])
        _, m_hidden = self.melody_encoder(melody, melody_mask)
        return m_hidden

    def getGT(self, melody):
        GT_melody = torch.zeros([melody.shape[0], 1]).long().to(device)
        bar_index = []
        melody_one = melody[0].cpu().tolist()
        chord_cnt = 1
        for i in range(len(melody_one)):
            if melody_one[i] >= 99:
                continue
            elif melody_one[i] == 0 and i != len(melody_one) - 1:
                bar_index.append(GT_melody.shape[1])
                GT_bar = torch.zeros([melody.shape[0], 1]).long().to(device)
                GT_melody = torch.cat((GT_melody, GT_bar), dim=1)
                chord_cnt += 2
            else:
                GT_melody = torch.cat((GT_melody, melody.narrow(1, i, 1)), dim=1)
        return GT_melody[:, 2:], bar_index

    def decoder(self, x, memory,bar_index, z_p, S_B):
        GT_x=x.detach()
        x_embed=self.melody_embedd(self.melody_linear(self.melody_input(x)))
        history=None
        z_list=[]
        time_cnt=0
        for i in range(x.shape[1]):
            if x[0,i]>=99:
                time_cnt+=1
        new_x = torch.zeros([x.shape[0], x.shape[1]-time_cnt-1, self.decoder_input]).to(device)
        tgt_mask = triple_mask(new_x.shape[0], new_x.shape[1]).to(device)
        bar_cnt=0;time_cnt=0
        for i in range(x.shape[1]):
            if x[0,i]>=99:
                timeSign=torch.nn.functional.one_hot(x[:,i]-99,num_classes=8).float().to(device)
                GT_x=torch.cat((GT_x[:,:(i-time_cnt)],GT_x[:,(i-time_cnt)+1:]),dim=-1)
                time_cnt+=1
            if x[0,i]==0:
                r_s=self.reverse_sentiment(S_B,bar_cnt)
                r_c=self.reverse_melody(x,bar_cnt,bar_index)
                b_mean,b_logv=self.B_recognitionNet(history, S_B[:, bar_cnt, :], r_s, r_c, timeSign)
                z_b = self.reparameterize(b_mean,b_logv)
                b_mean_p,b_logv_p=self.B_priorNet(history, S_B[:,bar_cnt,:],timeSign)
                z_list.append([b_mean,b_logv,b_mean_p,b_logv_p])
                j=i
                while True:
                    temp_x = torch.cat((x_embed[:, j, :], z_b, z_p, S_B[:, bar_cnt, :], timeSign), -1)
                    temp_x = LayerNorm(self.latent_size * 2 + self.c_size + 5 + 8).to(device)(temp_x)
                    new_x[:, j - time_cnt, :] = self.decoder_input_dim_reduct(temp_x)
                    j+=1
                    if j==x.shape[1]-1 or x[0,j]==0 or x[0,j]>=99:
                        break
                temp_x=self.m_decoder_layer1(new_x,tgt_mask)
                history=temp_x[:,j-1-time_cnt].detach()
                bar_cnt+=1
        x = temp_x
        for i in range(self.N):
            x = self.m_decoder_layers_2_chord[i](x, memory,full_mask(x.shape[0], x.shape[1], memory.shape[1]).to(device))
            if i < self.N - 1:
                x = self.m_decoder_layers_31[i](x, tgt_mask)
        x = self.m_decoder_layer4(x)
        x = self.m_decoder_norm(x)
        return x,z_list

    def forward(self,chord,melody,s_p,S_B):
        GT_note,bar_index=self.getGT(melody)
        chord_mask = full_mask(chord.shape[0], chord.shape[1], chord.shape[1])
        memory,chord_hidden = self.chord_encoder(chord,chord_mask)
        memory = self.memory_dim_add(memory)
        melody_mask = full_mask(melody.shape[0], melody.shape[1], melody.shape[1])
        _,melody_hidden = self.melody_encoder(melody,melody_mask)
        pValence_hidden = self.pValence_encoder(s_p)
        mean, logv = self.P_recognitionNet(chord_hidden, melody_hidden, pValence_hidden)
        mean_p, logv_p = self.P_priorNet(chord_hidden, pValence_hidden)
        KL_loss = -0.5 * torch.sum(1 + 2 * (logv - logv_p) - ((logv.exp().pow(2) + (mean_p - mean).pow(2)).div(logv_p.exp().pow(2))),
            1).mean()
        z_p = self.reparameterize(mean,logv)
        x,z_list = self.decoder(melody, memory,bar_index,z_p, S_B)
        b_KL_loss=0
        for i in z_list:
            b_KL_loss += -0.5 * torch.sum(
                1 + 2 * (i[1] - i[3]) - ((i[1].exp().pow(2) + (i[2] - i[0]).pow(2)).div(i[3].exp().pow(2))),
                1).mean()
        b_KL_loss=b_KL_loss/len(z_list)
        note=self.softmax(self.melody_note(x)+1e-10)
        return KL_loss,b_KL_loss,note,GT_note.detach()

    def getNote(self,flag,note,i):
        topk,topi=note.topk(10)
        if flag==1:
            for k in range(topi.shape[2]):
                if topi[:,i,k].squeeze(0).item()<=61:
                    return topi[:,i,k].unsqueeze(-1)
            print("wrong1")
        elif flag == 2:
            if topi[:, i, 0].squeeze(0).item() == 0:
                return topi[:, i, 0].unsqueeze(-1)
            else:
                return torch.LongTensor([[1]]).to(device)
        else:
            for k in range(topi.shape[2]):
                if topi[:,i,k].squeeze(0).item()>61:
                    return topi[:,i,k].unsqueeze(-1)
            print("wrong2")

    def generate(self,chord,s_p,S_B,timeSign):
        with torch.no_grad():
            chord_mask = full_mask(chord.shape[0], chord.shape[1], chord.shape[1])
            memory, chord_hidden = self.chord_encoder(chord, chord_mask)
            pValence_hidden = self.pValence_encoder(s_p)
            mean,logv=self.P_priorNet(chord_hidden, pValence_hidden)
            z_p = self.reparameterize(mean,logv)
            barNum=chord.shape[1]//2
            bar_cnt=0
            generate_notes = []
            input_x=torch.zeros([chord.shape[0],1]).long().to(device)
            new_x=torch.zeros([chord.shape[0],1,self.decoder_input]).to(device)
            memory = self.memory_dim_add(memory)
            i=0
            while True:
                if bar_cnt == 0:
                    mean,logv=self.B_priorNet(None, S_B[:, bar_cnt, :], timeSign)
                    z_b = self.reparameterize(mean,logv)
                else:
                    mean,logv=self.B_priorNet(history, S_B[:, bar_cnt, :],timeSign)
                    z_b = self.reparameterize(mean,logv)
                flag=1
                while True:
                    if input_x.shape[1]>2000:
                        return 0
                    x = self.melody_embedd(self.melody_linear(self.melody_input(input_x)))
                    tgt_mask=triple_mask(x.shape[0],x.shape[1])
                    temp_x = torch.cat((x[:, i, :], z_b, z_p, S_B[:, bar_cnt, :], timeSign), -1)
                    temp_x = LayerNorm(self.latent_size * 2 + self.c_size + 5 + 8).to(device)(temp_x)
                    new_x[:, i, :] = self.decoder_input_dim_reduct(temp_x)
                    x = self.m_decoder_layer1(new_x,tgt_mask)
                    history=x[:,-1,:]
                    for k in range(self.N):
                        x = self.m_decoder_layers_2_chord[k](x, memory,
                                                           full_mask(x.shape[0], x.shape[1], memory.shape[1]).to(device))
                        if k < self.N - 1:
                            x = self.m_decoder_layers_31[k](x, tgt_mask)
                    x = self.m_decoder_layer4(x)
                    x = self.m_decoder_norm(x)
                    note = self.softmax(self.melody_note(x))
                    topi=self.getNote(flag,note,i)
                    # print(topi)
                    note=topi.squeeze(0).item()
                    if note==0:
                        flag=1
                    elif note == 1 and i > 5:
                        flag = 2
                    else:
                        flag=flag*-1
                    generate_notes.append(note)
                    input_x=torch.cat([input_x,topi],dim=1)
                    temp = torch.zeros([chord.shape[0], 1, self.decoder_input]).to(device)
                    new_x=torch.cat([new_x,temp],dim=1)
                    i+=1
                    if note==0:
                        break
                bar_cnt+=1
                if bar_cnt==barNum:
                    break
            return generate_notes
