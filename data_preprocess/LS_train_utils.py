"""
Created on Mon Nov 20 20:03:00 2023
@author: Shulei Ji
# refer to https://github.com/melkor169/LeadSheetGen_Valence/blob/main/Preprocessing-Training/train_utils.py
"""
from statistics import median


def calc_chords_val(chords_bar):
    
    val = []
    
    for c in chords_bar:
        if 'maj7' in c: #maj 7th
            val.append(0.83)#2
        elif 'm7' in c: #minor 7th
            val.append(-0.46)#-1
        elif '7' in c: #major 7th
            val.append(-0.02)#0
        elif 'm9' in c: #minor 9th
            val.append(-0.15)#0
        elif '9' in c: #major 9th
            val.append(0.51)#1
        elif 'dim' in c: #diminished
            val.append(-0.43)#-1
        elif 'rest' in c: #Rest
            val.append(0)
        elif 'maj' in c: #Major
            val.append(0.87)#-2
        else: #Minor
            val.append(-0.81)#2
            
    #get the median
    med_val = median(val)
    # med_val = np.mean(val)
    
    #check the range
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
            
    return val_idx#round(val/len(chords_bar))


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
