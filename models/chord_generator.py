import torch
import torch.nn as nn
from models.MusicTV import MusicTV,full_mask,triple_mask

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


class chordVAE(MusicTV):
    def __init__(self,N,h,m_size,c_size,d_ff,hidden_size,latent_size,dropout):
        super().__init__(N,h,m_size,c_size,d_ff,hidden_size,latent_size,dropout)

        self.P_hidden2mean1 = nn.Linear(c_size * 2, latent_size)
        self.P_hidden2logv1 = nn.Linear(c_size * 2, latent_size)
        self.B_hidden2mean1 = nn.Linear(self.decoder_input + hidden_size * 2 + 5, latent_size)
        self.B_hidden2logv1 = nn.Linear(self.decoder_input + hidden_size * 2 + 5, latent_size)
        self.B_hidden2mean2 = nn.Linear(self.decoder_input + 5, latent_size)
        self.B_hidden2logv2 = nn.Linear(self.decoder_input + 5, latent_size)
        self.decoder_input_dim_reduct= nn.Linear(self.latent_size * 2 + c_size + 5, self.decoder_input)

    def P_priorNet(self,pValence_hidden):
        mean = self.P_hidden2mean2(pValence_hidden)
        logv = self.P_hidden2logv2(pValence_hidden)
        return mean, logv

    def P_recognitionNet(self,chord_hidden,s_p_hidden):
        concat = torch.cat((chord_hidden,s_p_hidden), -1)
        mean = self.P_hidden2mean1(concat)
        logv = self.P_hidden2logv1(concat)
        return mean, logv

    def B_recognitionNet(self,history,s_b,r_s,r_l):
        if history is None:
            history=torch.zeros([s_b.shape[0],self.decoder_input]).to(device)
        concat = torch.cat((history, s_b, r_s, r_l), -1)  # 64*1029  128+525
        mean = self.B_hidden2mean1(concat)
        logv = self.B_hidden2logv1(concat)
        return mean, logv

    def B_priorNet(self,history,s_b):
        if history is None:
            history = torch.zeros([s_b.shape[0], self.decoder_input]).to(device)
        concat = torch.cat((history, s_b), -1)
        mean = self.B_hidden2mean2(concat)
        logv = self.B_hidden2logv2(concat)
        return mean, logv

    def reverse_sentiment(self,x,i):
        reversed_s=torch.flip(x[:,i//2:,:],dims=[1])
        output,hidden=self.sentiment_lstm(reversed_s,None)
        hidden=hidden[0].mean(dim=0)
        return hidden

    def reverse_chord(self, chord, bar_cnt, bar_index=None):
        chord = chord[:, bar_cnt:, :]
        chord_mask = full_mask(chord.shape[0], chord.shape[1], chord.shape[1])
        _, c_hidden = self.chord_encoder(chord, chord_mask)
        return c_hidden

    def getGT(self, melody):
        melody_pad = torch.zeros([melody.shape[0], 1]).long().to(device)
        melody = torch.cat([melody, melody_pad], dim=-1)
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
        return bar_index

    def decoder(self, x, z_p, S_B):
        x_embed=self.chord_embedd(x)
        history=None
        z_list=[]
        new_x=torch.zeros([x.shape[0],x.shape[1]-1,self.decoder_input]).to(device)
        tgt_mask = triple_mask(new_x.shape[0], new_x.shape[1]).to(device)
        for i in range(x.shape[1]-1):
            if i%2==0:
                r_s=self.reverse_sentiment(S_B,i)
                r_c=self.reverse_chord(x,i)
                r_c = self.reverse_linear1(r_c)
                b_mean,b_logv=self.B_recognitionNet(history, S_B[:, i // 2, :], r_s, r_c)
                z_b = self.reparameterize(b_mean,b_logv)
                b_mean_p,b_logv_p=self.B_priorNet(history, S_B[:,i//2,:])
                z_list.append([b_mean,b_logv,b_mean_p,b_logv_p])
                temp_x=torch.cat((x_embed[:,i,:],z_b,z_p,S_B[:, i // 2, :]),-1)
                new_x[:, i, :] = self.decoder_input_dim_reduct(temp_x)
                temp_x = torch.cat((x_embed[:, i+1, :], z_b, z_p, S_B[:, i // 2, :]), -1)
                new_x[:, i+1, :] = self.decoder_input_dim_reduct(temp_x)
                temp_x=self.c_decoder_layer1(new_x,tgt_mask)
                history=temp_x[:,i+1,:].detach()
        x = temp_x
        for i in range(self.N):
            if i < self.N - 1:
                x = self.c_decoder_layers_31[i](x, tgt_mask)
        x = self.c_decoder_layer4(x)
        x = self.c_decoder_norm(x)
        return x,z_list

    def forward(self,chord,s_p,S_B):
        encode_chord=chord.narrow(1,1,chord.shape[1]-1)
        chord_mask=full_mask(encode_chord.shape[0],encode_chord.shape[1],encode_chord.shape[1]).to(device)
        _,chord_hidden = self.chord_encoder(encode_chord,chord_mask)
        pValence_hidden = self.pValence_encoder(s_p)
        mean, logv = self.P_recognitionNet(chord_hidden, pValence_hidden)
        mean_p, logv_p = self.P_priorNet(pValence_hidden)
        KL_loss = -0.5 * torch.sum(1 + 2 * (logv - logv_p) - ((logv.exp().pow(2) + (mean_p - mean).pow(2)).div(logv_p.exp().pow(2))),
            1).mean()
        z_p = self.reparameterize(mean,logv)
        x,z_list = self.decoder(chord, z_p, S_B)
        b_KL_loss=0
        for i in z_list:
            b_KL_loss += -0.5 * torch.sum(
                1 + 2 * (i[1] - i[3]) - ((i[1].exp().pow(2) + (i[2] - i[0]).pow(2)).div(i[3].exp().pow(2))),
                1).mean()
        b_KL_loss=b_KL_loss/len(z_list)
        root=self.softmax(self.chord_root(x)+1e-10)
        type=self.softmax(self.chord_type(x)+1e-10)
        return KL_loss,b_KL_loss,root,type

    def generate(self,s_p,S_B):
        with torch.no_grad():
            len =S_B.shape[1]
            pValence_hidden = self.pValence_encoder(s_p)
            mean,logv=self.P_priorNet(pValence_hidden)
            z_p = self.reparameterize(mean,logv)
            history = []
            x_pad=torch.zeros([S_B.shape[0],len*2+1,48]).to(device)
            x_pad[:,:,0]=1;x_pad[:,:,-1]=1
            new_x=torch.zeros([S_B.shape[0],len*2,self.decoder_input]).to(device)
            tgt_mask=triple_mask(new_x.shape[0],new_x.shape[1])
            for i in range(len*2):
                x=self.chord_embedd(x_pad)
                if i % 2 == 0:
                    if i==0:
                        mean,logv=self.B_priorNet(None, S_B[:, i // 2, :])
                        z_b = self.reparameterize(mean,logv)
                    else:
                        mean,logv=self.B_priorNet(history, S_B[:, i // 2, :])
                        z_b = self.reparameterize(mean,logv)
                new_x[:, i, :] = self.decoder_input_dim_reduct(torch.cat((x[:, i, :], z_b, z_p, S_B[:, i // 2, :]), -1))
                x = self.c_decoder_layer1(new_x,tgt_mask)
                history=x[:,i,:]
                for k in range(self.N):
                    if k < self.N - 1:
                        x = self.c_decoder_layers_31[k](x, tgt_mask)
                x = self.c_decoder_layer4(x)
                x = self.c_decoder_norm(x)
                root = self.softmax(self.chord_root(x))
                type = self.softmax(self.chord_type(x))
                x_next=self.softmax2chord(type,root)
                x_pad[:,i+1,:]=x_next[:,i,:]
        return x_pad[:,1:,:]
