"""
Created on Fri Oct 29 17:28:18 2022
@author: Shulei Ji
"""

import numpy as np
import torch

def calc_chords_val(chords_bar):

    val = []

    for c in chords_bar:
        if 'maj7' in c:  # maj 7th
            val.append(0.83)  # 2
        elif 'm7' in c:  # minor 7th
            val.append(-0.46)  # -1
        elif '7' in c:  # major 7th
            val.append(-0.02)  # 0
        elif 'm9' in c:  # minor 9th
            val.append(-0.15)  # 0
        elif '9' in c:  # major 9th
            val.append(0.51)  # 1
        elif 'dim' in c:  # diminished
            val.append(-0.43)  # -1
        elif 'rest' in c:  # Rest
            val.append(0)
        elif 'maj' in c:  # Major
            val.append(0.87)  # -2
        else:  # Minor
            val.append(-0.81)  # 2

    # get the median
    med_val = np.mean(val)

    # check the range
    if med_val > 0.6:
        val_idx = 2
    elif med_val > 0.2 and med_val <= 0.6:
        val_idx = 1
    elif med_val > -0.2 and med_val <= 0.2:
        val_idx = 0
    elif med_val > -0.6 and med_val <= -0.2:
        val_idx = -1
    else:
        val_idx = -2

    return val_idx

def calc_piece_val(valenceSeq):
    score = sum(valenceSeq) / len(valenceSeq)
    if score >= 1.5:
        cat = 2
    elif score >= 0.5 and score < 1.5:
        cat = 1
    elif score >= -0.5 and score < 0.5:
        cat = 0
    elif score >= -1.5 and score < -0.5:
        cat = -1
    else:
        cat = -2
    return cat

def bar_padding(melody):
    for i in range(len(melody)):
        last = melody[i].index(0)
        new_data_i = melody[i][:last]
        for j in range(last + 1, len(melody[i])):
            if melody[i][j] == 0:
                new_data_i.extend(melody[i][last:j])
                for k in range(42 - (j - last - 1)):
                    new_data_i.append(1)
                last = j
        new_data_i.extend(melody[i][last:])
        for k in range(42 - (len(melody[i]) - last - 1)):
            new_data_i.append(1)
        melody[i]=new_data_i
    return melody


def list2tensor(melody,chord,valence,device):
    new_melody=[]
    new_chord=[]
    new_valence=[]
    for i in range(len(melody)):
        if i%12==0:
            melody_temp = torch.LongTensor(melody[i:i + 12]).to(device)
            chord_temp = torch.LongTensor(chord[i:i + 12]).to(device)
            valence_temp = (torch.LongTensor(valence[i:i + 12]) + 2).to(device)
            new_melody.append(melody_temp)
            new_chord.append(chord_temp)
            new_valence.append(valence_temp)
    return new_melody,new_chord,new_valence

