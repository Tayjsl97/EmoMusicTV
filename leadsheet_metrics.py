"""
Created on Fri Oct 29 17:28:18 2022
@author: Shulei Ji
"""

from chord_metrics import chordRepre2interval,tonal_centroid,halfBar_sequences
import numpy as np

TIMESIGN={0:[6, 8], 1:[4, 4], 2:[9, 8], 3:[2, 4], 4:[3, 4], 5:[2, 2], 6:[6, 4], 7:[3, 2]}
DURATION={0:2, 1:4, 2:6, 3:8, 4:10, 5:12, 6:16, 7:18, 8:20, 9:22, 10:24, 11:30, 12:32, 13:36, 14:42, 15:44, 16:48, 17:54, 18:56, 19:60,
20:64, 21:66, 22:68, 23:72, 24:78, 25:80, 26:84, 27:90, 28:92, 29:96, 30:102, 31:108, 32:120, 33:126, 34:132, 35:138, 36:144}

def compute_metrics(chord_sequence,melody_sequence):
    chord_type = {}
    chord_type[0] = 0
    for i in range(1, 7):
        for j in range(40):
            chord_type[i * 100 + j] = 0
    for i in chord_sequence:
        # print(i)
        type = i[:7].index(1)
        root = i[7:].index(1)
        if type != 0 and root!=40:
            chord_type[type * 100 + root] += 1
        else:
            chord_type[0] += 1
    ########################## CC & CHE ############################
    chord_statistics = []
    for i in chord_type:
        chord_statistics.append(chord_type[i])
    CC = len(chord_statistics) - chord_statistics.count(0)
    # # calculate entropy
    chord_statistics = np.array(chord_statistics) / len(chord_statistics)
    CHE = sum([- p_i * np.log(p_i + 1e-6) for p_i in chord_statistics])
    ############################# CTD ############################
    y = 0
    CTD_length = len(chord_sequence) - 1
    for n in range(len(chord_sequence) - 1):
        chord1 = chordRepre2interval(chord_sequence[n + 1])
        chord2 = chordRepre2interval(chord_sequence[n])
        if len(chord1) > 1 and len(chord2) > 1:
            y += np.sqrt(
                np.sum((np.asarray(tonal_centroid(chord1))
                        - np.asarray(tonal_centroid(chord2))) ** 2))
        else:
            CTD_length -= 1
    if CTD_length == 0:
        CTD = 1
    else:
        CTD = y / CTD_length
    ################### PCHE & DCHE & API #######################
    pitch = []
    pitch_class = np.array([0] * 12)
    duration_class = np.array([0] * 37)
    restNum = 0;
    pitchNum = 0;
    durationNum = 0
    for i in melody_sequence:
        if i in [0, 99, 100, 101, 102, 103, 104, 105, 106]:
            continue
        if i <= 61:
            if i == 1:
                restNum += 1
            else:
                pitch.append((i + 40) % 12)
                pitch_class[(i + 40) % 12] += 1
                pitchNum += 1
        else:
            duration_class[i - 62] += 1
            durationNum += 1
    pitch_class = pitch_class / pitchNum
    PCHE = sum([- p_i * np.log(p_i + 1e-6) for p_i in pitch_class])
    duration_class = duration_class / durationNum
    DCHE = sum([- p_i * np.log(p_i + 1e-6) for p_i in duration_class])
    intervalSum = 0
    for i in range(len(pitch) - 1):
        intervalSum += abs(pitch[i + 1] - pitch[i])
    API = intervalSum / (len(pitch) - 1)
    ##########################################################
    chord_zip, melodyPit_zip, melodyDur_zip = halfBar_sequences(chord_sequence, melody_sequence)
    # print(len(chord_zip), len(melodyPit_zip), len(melodyDur_zip))
    assert len(chord_zip) == len(melodyPit_zip) == len(melodyDur_zip)
    ########################## HC ############################
    HC_note_cnt=0
    for melodyPit_m, chord_m in zip(melodyPit_zip, chord_zip):
        m_i = 0
        for i in range(len(melodyPit_m)):
            m = melodyPit_m[i]
            if m in chord_m:
                m_i += 1
        HC_note_cnt+=len(melodyPit_m)
    HC = m_i / HC_note_cnt
    ########################## DC ############################
    DC = 0
    for melodyPit_m, chord_m in zip(melodyPit_zip, chord_zip):
        m_i = 0
        for i in range(len(melodyPit_m)):
            m = melodyPit_m[i]
            if m not in chord_m:
                m_i += 1
        if m_i == len(melodyPit_m):
            DC += 1
    DC = DC / len(chord_sequence)
    ############################# CTnCTR ############################
    c = 0
    p = 0
    n = 0
    for melodyPit_m, melodyDur_m, chord_m in zip(melodyPit_zip, melodyDur_zip, chord_zip):
        if len(chord_m) == 1:
            for ii in melodyPit_m:
                if ii != -1:
                    c += 1
            continue
        for i in range(len(melodyPit_m)):
            m = melodyPit_m[i]
            if m != -1:
                if m in chord_m:
                    c += melodyDur_m[i]
                else:
                    n += melodyDur_m[i]
                    if i + 1 < len(melodyPit_m):
                        if abs(melodyPit_m[i + 1] - melodyPit_m[i]) <= 2:
                            p += melodyDur_m[i]
    if (c + n) == 0:
        CTnCTR = 1
    else:
        CTnCTR = (c + p) / (c + n)
    ############################# PCS ############################
    score = 0
    count = 0
    for melodyPit_m, melodyDur_m, chord_m in zip(melodyPit_zip, melodyDur_zip, chord_zip):
        if len(chord_m) == 1:
            continue
        # print("melody_m: ",melody_m)
        for m in range(len(melodyPit_m)):
            if melodyPit_m[m] != -1:
                score_m = 0
                for c in chord_m:
                    # unison, maj, minor 3rd, perfect 5th, maj, minor 6,
                    if abs(melodyPit_m[m] - c) == 0 or abs(melodyPit_m[m] - c) == 3 or abs(melodyPit_m[m] - c) == 4 \
                            or abs(melodyPit_m[m] - c) == 7 or abs(melodyPit_m[m] - c) == 8 or abs(
                        melodyPit_m[m] - c) == 9 \
                            or abs(melodyPit_m[m] - c) == 5:
                        if (melodyPit_m[m] - c+12)%12 == 5:
                            score_m += 0
                        else:
                            score_m += 1
                    else:
                        score_m += -1
                # print(m,duration_m[m])
                score += score_m * melodyDur_m[m]
                count += melodyDur_m[m]
    if count == 0:
        PCS = 1
    else:
        PCS = score / count
    ############################# MCTD ############################
    y = 0
    y_m = 0
    count = 0
    for melodyPit_m, melodyDur_m, chord_m in zip(melodyPit_zip, melodyDur_zip, chord_zip):
        if len(chord_m) == 1:
            continue
        for m in range(len(melodyPit_m)):
            if melodyPit_m[m] != -1:
                y += np.sqrt(
                    np.sum((np.asarray(tonal_centroid([melodyPit_m[m]])) - np.asarray(tonal_centroid(chord_m)))) ** 2)
                y_m += y * melodyDur_m[m]
                count += melodyDur_m[m]
    if count == 0:
        MCTD = 1
    else:
        MCTD = y / count
    return PCHE,DCHE,API,HC,CC,CHE,CTD,DC,CTnCTR,PCS,MCTD