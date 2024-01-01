"""
Created on Fri Oct 29 17:28:18 2022
@author: Shulei Ji
"""

import torch
import torch.nn as nn
import random
import pickle
import os
import time
import math
import datetime
import numpy as np
from pytorchtools import EarlyStopping
from models.melodyVAE_givenHarmony import melodyVAE
from utils import list2tensor,bar_padding


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device=torch.device("cpu")
print(torch.cuda.is_available())
print("device: ",device)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(melody,chord,valence,step,allStep):
    coefficient = (1 / allStep) * step*0.05
    P_coefficient=(1 / allStep) * step*0.001
    s_p=torch.nn.functional.one_hot(valence.narrow(1,0,1),num_classes=5).float()
    S_B=torch.nn.functional.one_hot(valence.narrow(1,1,valence.shape[1]-1),num_classes=5).float()
    KL_loss,b_KL_loss,note,GT_notes=VAE(chord,melody,s_p,S_B)
    l = 0
    for k in range(12):
        l += criterion(note[k], GT_notes[k])
    reconstruction_loss = l/12
    Piece_KL_loss = P_coefficient * KL_loss
    Bar_KL_loss = coefficient * b_KL_loss
    loss_update=reconstruction_loss+Piece_KL_loss+Bar_KL_loss
    loss_record=reconstruction_loss+KL_loss+b_KL_loss
    return loss_update,loss_record.item(),reconstruction_loss.item(),KL_loss.item(),b_KL_loss.item()


start_time=time.time()
def timeSince(since):
    now=time.time()
    s=now-since
    h=math.floor(s/3600)
    s-=h*3600
    m=math.floor(s/60)
    s-=m*60

    return '%dh_%dm_%ds' % (h, m, s)


def trainIter(train_melody,train_chord,train_valence,test_melody,test_chord,test_valence,Epoch):
    model_path = "./save_models/melodyVAE_scratch_"+dataset +"/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    log_path = './logs/melodyVAE_scratch_'+dataset +'.log'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    f = open(log_path, 'a')
    f.write('\nbatch_size: %.6d lr: %.6f' % (batch_size,learning_rate))
    f.close()
    train_length = len(train_melody)
    test_length = len(test_melody)
    max_test_loss = 1000
    lr = learning_rate
    step=0
    allStep=train_length*100
    # epoch_already=dict['epoch']
    # step = (train_length // batch_size) * epoch_already
    for epoch in range(0,Epoch):
        f = open(log_path, 'a')
        print("-----------------------------epoch ", epoch, "------------------------------")
        f.write('\n-----------------------------epoch %d------------------------------' % (epoch))
        train_total_loss=0
        rl_total=0;kl_total=0;P_kl_total=0
        temp_total_loss = 0;temp_rl_loss = 0
        temp_Pkl = 0;temp_kl = 0
        train_start_idx=0
        VAE.train()
        i=0
        while i < train_length:
            loss_update = 0
            mem=0
            for j in range(batch_size // 12):
                if i+j>=train_length:
                    break
                melody,chord,valence=train_melody[i+j],train_chord[i+j],train_valence[i+j]
                if melody.shape[1]>600:
                    continue
                mem += melody.shape[1]
                if mem > 600:
                    optimizer.zero_grad()
                    loss_update.backward()
                    nn.utils.clip_grad_norm_(VAE.parameters(), 2)
                    optimizer.step()
                    loss_update = 0
                loss_back,loss,rl,P_kl,kl=train(melody,chord,valence,step,allStep)
                step+=1
                loss_update+=loss_back
                temp_total_loss+=loss*12
                train_total_loss+=loss*12
                temp_rl_loss+=rl*12
                rl_total+=rl*12;
                kl_total+=kl*12;P_kl_total+=P_kl*12
                temp_kl += kl * 12;temp_Pkl += P_kl * 12
                train_start_idx+=12
            i=i+j+1
            if loss_update!=0:
                optimizer.zero_grad()
                loss_update.backward()
                nn.utils.clip_grad_norm_(VAE.parameters(), 2)
                optimizer.step()
            if train_start_idx!=0 and temp_total_loss!=0 and train_start_idx%(print_every)==0:
                print('epoch train:%d, %s(%d %d%%) total: %.6f rl: %.6f Pkl: %.6f kl: %.6f' % (
                    epoch, timeSince(start_time), train_start_idx,
                    train_start_idx / (train_length * 12) * 100,
                    temp_total_loss / print_every, temp_rl_loss / print_every, temp_Pkl / print_every,
                    temp_kl / print_every))
                f.write('\nepoch train:%d, %s(%d %d%%) total: %.6f rl: %.6f Pkl: %.6f kl: %.6f' % (
                    epoch, timeSince(start_time), train_start_idx,
                    train_start_idx / (train_length * 12) * 100,
                    temp_total_loss / print_every, temp_rl_loss / print_every, temp_Pkl / print_every,
                    temp_kl / print_every))
                print("--------------------------------------------------------------")
                temp_total_loss = 0
                temp_rl_loss = 0
                temp_Pkl = 0
                temp_kl = 0
        test_start_idx = 0
        test_total_loss= 0
        rl_total_test=0;kl_total_test=0;P_kl_total_test=0
        VAE.eval()
        for i in range(test_length):
            melody, chord, valence = test_melody[i], test_chord[i], test_valence[i]
            if melody.shape[1]>600:
                continue
            _,loss,rl,P_kl,kl=train(melody,chord,valence,step,allStep)
            test_total_loss+=loss*test_batch_size
            rl_total_test+=rl*test_batch_size;kl_total_test+=kl*test_batch_size;P_kl_total_test+=P_kl*test_batch_size
            test_start_idx+=test_batch_size
        print('epoch: %d, time: %s, \ntrain loss: %.6f, test loss: %.6f, \nrl_train: %.6f, rl_test: %.6f, \n'
              'kl_train: %.6f,kl_test: %.6f, \nP_kl_train: %.6f, P_kl_test: %.6f, \nlearning rate: %.6f'
              % (epoch, timeSince(start_time), train_total_loss / train_start_idx,test_total_loss/test_start_idx,
                 rl_total/train_start_idx,rl_total_test/test_start_idx,kl_total/train_start_idx,kl_total_test/test_start_idx,
                 P_kl_total/train_start_idx,P_kl_total_test/test_start_idx,lr))
        f.write('\nepoch: %d, time: %s, \ntrain loss: %.6f, test loss: %.6f, \nrl_train: %.6f, rl_test: %.6f, \n'
              'kl_train: %.6f,kl_test: %.6f, \nP_kl_train: %.6f, P_kl_test: %.6f, \nlearning rate: %.6f'
              % (epoch, timeSince(start_time), train_total_loss / train_start_idx,test_total_loss/test_start_idx,
                 rl_total/train_start_idx,rl_total_test/test_start_idx,kl_total/train_start_idx,kl_total_test/test_start_idx,
                 P_kl_total/train_start_idx,P_kl_total_test/test_start_idx,lr))
        train_average = train_total_loss / train_start_idx
        test_average = test_total_loss / test_start_idx
        early_stopping(test_average, VAE)
        if early_stopping.early_stop:
            print("------------Early Stopping-------------")
            break
        if test_average < max_test_loss:
            print("epoch: %d save min test loss model-->test loss: %.6f" % (epoch, test_average))
            f.write('\nepoch: %d save min test loss model-->test loss: %.6f' % (epoch, test_average))
            model_save_path = model_path + "melodyVAE_" + dataset + "_epoch" + str(epoch) + "_min_" + str(round(test_average,4)) + ".pth"
            state = {'model': VAE.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, model_save_path)
            max_test_loss = test_average
        else:
            if epoch%5==0:
                model_save_path = model_path + "melodyVAE_" + dataset + "_epoch" + str(epoch) + "_" + str(round(test_average,4)) + ".pth"
                state = {'model': VAE.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state, model_save_path)
        f.close()


if __name__ == '__main__':
    ################################### configuration
    setup_seed(13)
    dataset = "NMD"
    date = str(datetime.date.today())
    batch_size = 24
    test_batch_size=12
    patience = 20
    Epoch = 200
    VAE = melodyVAE(N=3, h=4, m_size=8, c_size=48, d_ff=256, hidden_size=256, latent_size=128, dropout=0.2).to(device)
    criterion = nn.NLLLoss().to(device)
    learning_rate = 3e-4
    optimizer = torch.optim.Adam(VAE.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience, verbose=True)
    print_every = 1020
    ################################# extract data
    file = open("./data/" + dataset + "_melody_train.data", 'rb')
    train_melody = pickle.load(file)
    file = open("./data/" + dataset + "_chord_train.data", 'rb')
    train_chord = pickle.load(file)
    file = open("./data/" + dataset + "_chord_train_valence.data", 'rb')
    train_valence = pickle.load(file)
    print(len(train_melody), len(train_chord), len(train_valence))
    file = open("./data/" + dataset + "_melody_test.data", 'rb')
    test_melody = pickle.load(file)
    file = open("./data/" + dataset + "_chord_test.data", 'rb')
    test_chord = pickle.load(file)
    file = open("./data/" + dataset + "_chord_test_valence.data", 'rb')
    test_valence = pickle.load(file)
    print(len(test_melody), len(test_chord), len(test_valence))
    # train_melody=bar_padding(train_melody)  # padding each bar to the equal length
    train_melody,train_chord,train_valence=list2tensor(train_melody,train_chord,train_valence,device)
    print(len(train_melody), len(train_chord), len(train_valence))
    test_melody, test_chord, test_valence = list2tensor(test_melody, test_chord, test_valence,device)
    print(len(test_melody), len(test_chord), len(test_valence))
    ################################# begin train
    trainIter(train_melody,train_chord,train_valence,test_melody,test_chord,test_valence,Epoch)
