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
from models.MusicTV import MusicTV

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print("device: ",device)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(melody_pre,melody_post,chord_pre,chord_post,valence,step,allStep):
    coefficient = (1 / allStep) * step*0.1
    P_coefficient=(1 / allStep) * step*0.005
    s_p=torch.nn.functional.one_hot(valence.narrow(1,0,1),num_classes=5).float()
    S_B=torch.nn.functional.one_hot(valence.narrow(1,13,4),num_classes=5).float()
    KL_loss, b_KL_loss, note, type, root, GT_note, GT_type, GT_root=VAE(melody_pre,melody_post,chord_pre,chord_post,s_p,S_B)
    l_melody = 0
    l_chord = 0
    for k in range(12):
        l_melody += criterion(note[k], GT_note[k])
        l_chord += criterion(type[k], GT_type[k])
        l_chord += criterion(root[k], GT_root[k])
    reconstruction_loss = (l_melody+l_chord)/12
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
    model_path = "./save_models/MusicTV_124/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    f = open('./logs/MusicTV_124.log', 'a')
    f.write('\nbatch_size: %.6d lr: %.6f' % (batch_size,learning_rate))
    f.close()
    train_length = len(train_melody)
    test_length = len(test_melody)
    max_test_loss = 1000
    lr = learning_rate
    step=0
    allStep=(train_length//12)*100
    # epoch_already=dict['epoch']
    # step = (train_length // 12) * epoch_already
    for epoch in range(0,Epoch):
        f = open('./logs/MusicTV_124.log', 'a')
        print("-----------------------------epoch ", epoch, "------------------------------")
        f.write('\n-----------------------------epoch %d------------------------------' % (epoch))
        train_start_idx=0
        train_total_loss=0
        rl_total=0;kl_total=0;P_kl_total=0
        temp_total_loss = 0;temp_rl_loss = 0
        temp_Pkl=0;temp_kl=0
        VAE.train()
        for i in range(train_length):
            if i+batch_size>train_length:
                break
            if i%batch_size!=0:
                continue
            loss_update=0
            for j in range(batch_size//12):
                melody,chord,valence=train_melody[i+12*j:i+12*(j+1)],train_chord[i+12*j:i+12*(j+1)],train_valence[i+12*j:i+12*(j+1)]
                melody_pre=torch.LongTensor([i[0] for i in melody]).to(device)
                melody_post=torch.LongTensor([i[1] for i in melody]).to(device)
                chord_pre=torch.Tensor([i[:24] for i in chord]).to(device)
                chord_post = torch.Tensor([i[24:32] for i in chord]).to(device)
                valence=(torch.LongTensor(valence)+2).to(device)
                melody_pad = torch.zeros([melody_pre.shape[0], 1]).long().to(device)
                melody_post = torch.cat([melody_post, melody_pad], dim=-1)
                chord_pad = torch.zeros([chord_post.shape[0], 1,48]).long().to(device)
                chord_pad[:,:,0]=1;chord_pad[:,:,-1]=1
                chord_post=torch.cat([chord_pad, chord_post], dim=1)
                loss_back,loss,rl,P_kl,kl=train(melody_pre,melody_post,chord_pre,chord_post,valence,step,allStep)
                step+=1
                loss_update+=loss_back
                temp_total_loss+=loss*12
                train_total_loss+=loss*12
                temp_rl_loss+=rl*12
                temp_Pkl+=P_kl*12;temp_kl+=kl*12
                rl_total+=rl*12;
                kl_total+=kl*12;P_kl_total+=P_kl*12
                train_start_idx+=12
            optimizer.zero_grad()
            loss_update.backward()
            nn.utils.clip_grad_norm_(VAE.parameters(), 2)
            optimizer.step()
            if train_start_idx%(print_every)==0:
                print('epoch train:%d, %s(%d %d%%) total: %.6f rl: %.6f Pkl: %.6f kl: %.6f' % (
                    epoch, timeSince(start_time), train_start_idx, train_start_idx / ((train_length // batch_size) * batch_size) * 100,
                    temp_total_loss/print_every,temp_rl_loss /print_every, temp_Pkl/print_every, temp_kl/print_every))
                f.write('\nepoch train:%d, %s(%d %d%%) total: %.6f rl: %.6f Pkl: %.6f kl: %.6f' % (
                    epoch, timeSince(start_time), train_start_idx, train_start_idx / ((train_length // batch_size) * batch_size) * 100,
                    temp_total_loss/print_every,temp_rl_loss /print_every, temp_Pkl/print_every, temp_kl/print_every))
                print("--------------------------------------------------------------")
                temp_total_loss = 0
                temp_rl_loss=0
                temp_Pkl=0;temp_kl=0
        test_start_idx = 0
        test_total_loss= 0
        rl_total_test=0;kl_total_test=0;P_kl_total_test=0
        VAE.eval()
        for i in range(test_length):
            if i + batch_size > test_length:
                break
            if i % batch_size != 0:
                continue
            melody, chord, valence = test_melody[i:i + batch_size], test_chord[i:i + batch_size], test_valence[i:i + batch_size]
            melody_pre = torch.LongTensor([i[0] for i in melody]).to(device)
            melody_post = torch.LongTensor([i[1] for i in melody]).to(device)
            chord_pre = torch.Tensor([i[:24] for i in chord]).to(device)
            chord_post = torch.Tensor([i[24:32] for i in chord]).to(device)
            valence = (torch.LongTensor(valence) + 2).to(device)
            melody_pad = torch.zeros([melody_pre.shape[0], 1]).long().to(device)
            melody_post = torch.cat([melody_post, melody_pad], dim=-1)
            chord_pad = torch.zeros([chord_post.shape[0], 1, 48]).long().to(device)
            chord_pad[:, :, 0] = 1;chord_pad[:, :, -1] = 1
            chord_post = torch.cat([chord_pad, chord_post], dim=1)
            _,loss, rl, P_kl, kl = train(melody_pre, melody_post, chord_pre, chord_post, valence, step, allStep)
            test_total_loss+=loss*batch_size
            rl_total_test+=rl*batch_size
            kl_total_test+=kl*batch_size
            P_kl_total_test+=P_kl*batch_size
            test_start_idx+=batch_size
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
            model_save_path = model_path + "MusicTV_124_epoch" + str(epoch) + "_min_" + str(round(test_average,4)) + ".pth"
            state = {'model': VAE.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, model_save_path)
            max_test_loss = test_average
        else:
            if epoch%5==0:
                model_save_path = model_path + "MusicTV_124_epoch" + str(epoch) + "_" + str(round(test_average,4)) + ".pth"
                state = {'model': VAE.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state, model_save_path)
        f.close()


if __name__ == '__main__':
    ################################### configuration
    setup_seed(13)
    date = str(datetime.date.today())
    batch_size = 72
    patience = 20
    Epoch = 200
    VAE = MusicTV(N=3,h=4,m_size=8,c_size=48,d_ff=256,hidden_size=256,latent_size=128,dropout=0.2).to(device)
    #resume="./save_models/MusicTV_124/MusicTV_124_epoch109_min_2.9357.pth"
    # dict=torch.load(resume)
    # VAE.load_state_dict(dict['model'])
    criterion = nn.NLLLoss().to(device)
    learning_rate = 3e-4
    optimizer=torch.optim.Adam(VAE.parameters(),lr=learning_rate)
    # optimizer.load_state_dict(dict['optimizer'])
    early_stopping = EarlyStopping(patience, verbose=True)
    print_every = 6012
    ################################# extract data
    file = open("./data/All_124_melody_train.data", 'rb')
    train_melody = pickle.load(file)
    file = open("./data/All_124_chord_train.data", 'rb')
    train_chord = pickle.load(file)
    file = open("./data/All_124_valence_train.data", 'rb')
    train_valence = pickle.load(file)
    print(len(train_melody),len(train_chord),len(train_valence))
    file = open("./data/All_124_melody_test.data", 'rb')
    test_melody = pickle.load(file)
    file = open("./data/All_124_chord_test.data", 'rb')
    test_chord = pickle.load(file)
    file = open("./data/All_124_valence_test.data", 'rb')
    test_valence = pickle.load(file)
    print(len(test_melody),len(test_chord),len(test_valence))
    ################################# begin train
    trainIter(train_melody,train_chord,train_valence,test_melody,test_chord,test_valence,Epoch)
