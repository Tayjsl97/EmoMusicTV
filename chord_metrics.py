"""
Created on Fri Oct 29 17:28:18 2022
@author: Shulei Ji
"""

import math
import numpy as np
import pickle

def tonal_centroid(notes):
    fifths_lookup = {9: [1.0, 0.0], 2: [math.cos(math.pi / 6.0), math.sin(math.pi / 6.0)],
                     7: [math.cos(2.0 * math.pi / 6.0), math.sin(2.0 * math.pi / 6.0)],
                     0: [0.0, 1.0], 5: [math.cos(4.0 * math.pi / 6.0), math.sin(4.0 * math.pi / 6.0)],
                     10: [math.cos(5.0 * math.pi / 6.0), math.sin(5.0 * math.pi / 6.0)],
                     3: [-1.0, 0.0], 8: [math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
                     1: [math.cos(8.0 * math.pi / 6.0), math.sin(8.0 * math.pi / 6.0)],
                     6: [0.0, -1.0], 11: [math.cos(10.0 * math.pi / 6.0), math.sin(10.0 * math.pi / 6.0)],
                     4: [math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)]}
    minor_thirds_lookup = {3: [1.0, 0.0], 7: [1.0, 0.0], 11: [1.0, 0.0],
                           0: [0.0, 1.0], 4: [0.0, 1.0], 8: [0.0, 1.0],
                           1: [-1.0, 0.0], 5: [-1.0, 0.0], 9: [-1.0, 0.0],
                           2: [0.0, -1.0], 6: [0.0, -1.0], 10: [0.0, -1.0]}
    major_thirds_lookup = {0: [0.0, 1.0], 3: [0.0, 1.0], 6: [0.0, 1.0], 9: [0.0, 1.0],
                           2: [math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
                           5: [math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
                           8: [math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
                           11: [math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
                           1: [math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)],
                           4: [math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)],
                           7: [math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)],
                           10: [math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)]}

    fifths = [0.0, 0.0]
    minor = [0.0, 0.0]
    major = [0.0, 0.0]
    r1 = 1
    r2 = 1
    r3 = 0.5
    if notes:
        for note in notes:
            note=note%12
            for i in range(2):
                fifths[i] += r1 * fifths_lookup[note][i]
                minor[i] += r2 * minor_thirds_lookup[note][i]
                major[i] += r3 * major_thirds_lookup[note][i]
        for i in range(2):
            fifths[i] /= len(notes)
            minor[i] /= len(notes)
            major[i] /= len(notes)

    return fifths + minor + major


def chordRepre2interval(chord):
    CHORDTYPE = {1: [3, 4], 2: [3, 3], 3: [4, 3], 4: [3, 4, 3], 5: [4, 3, 3], 6: [4, 3, 4]}
    type=chord[:8]
    if type.index(1) in [0,7]:
        return [0]
    else:
        return CHORDTYPE[type.index(1)]


CHORD = {0: '[0]', 1: '[50, 53, 56]', 2: '[36, 39, 43, 46]', 3: '[67, 70, 74]', 4: '[50, 53, 57, 60]', 5: '[61, 64, 67]', 6: '[32, 36, 39, 42]', 7: '[37, 40, 44, 47]', 8: '[49, 52, 56, 59]', 9: '[55, 59, 62]', 10: '[39, 42, 46, 49]', 11: '[30, 33, 37]', 12: '[41, 44, 48]', 13: '[36, 40, 43]', 14: '[37, 41, 44]', 15: '[47, 50, 54, 57]', 16: '[30, 34, 37, 40]', 17: '[57, 60, 64]', 18: '[49, 52, 56]', 19: '[63, 66, 70]', 20: '[36, 40, 43, 46]', 21: '[55, 58, 62, 65]', 22: '[59, 62, 66]', 23: '[66, 70, 73]', 24: '[55, 58, 62]', 25: '[34, 37, 40]', 26: '[44, 47, 51, 54]', 27: '[38, 41, 45]', 28: '[51, 54, 57]', 29: '[38, 41, 44]', 30: '[62, 65, 69]', 31: '[46, 50, 53, 57]', 32: '[38, 42, 45, 48]', 33: '[43, 47, 50, 53]', 34: '[44, 47, 50]', 35: '[63, 66, 70, 73]', 36: '[46, 49, 52]', 37: '[56, 60, 63, 67]', 38: '[33, 36, 40]', 39: '[44, 48, 51]', 40: '[35, 38, 42, 45]', 41: '[35, 39, 42, 46]', 42: '[37, 40, 43]', 43: '[32, 36, 39, 43]', 44: '[39, 42, 46]', 45: '[42, 45, 48]', 46: '[36, 40, 43, 47]', 47: '[44, 47, 51]', 48: '[58, 61, 64]', 49: '[57, 61, 64]', 50: '[41, 45, 48, 52]', 51: '[59, 62, 66, 69]', 52: '[61, 65, 68]', 53: '[38, 41, 45, 48]', 54: '[68, 71, 75]', 55: '[57, 61, 64, 67]', 56: '[42, 46, 49]', 57: '[33, 37, 40, 43]', 58: '[47, 51, 54, 57]', 59: '[40, 44, 47]', 60: '[36, 39, 42]', 61: '[64, 67, 70]', 62: '[55, 59, 62, 66]', 63: '[53, 56, 60, 63]', 64: '[60, 64, 67]', 65: '[39, 43, 46]', 66: '[63, 67, 70, 73]', 67: '[31, 35, 38, 41]', 68: '[41, 44, 48, 51]', 69: '[59, 63, 66, 69]', 70: '[32, 35, 38]', 71: '[43, 46, 50]', 72: '[46, 50, 53]', 73: '[49, 52, 55]', 74: '[52, 55, 59]', 75: '[45, 49, 52, 55]', 76: '[53, 57, 60, 64]', 77: '[68, 71, 74]', 78: '[65, 68, 71]', 79: '[65, 69, 72, 75]', 80: '[53, 57, 60]', 81: '[48, 52, 55, 58]', 82: '[63, 67, 70]', 83: '[61, 65, 68, 71]', 84: '[43, 46, 50, 53]', 85: '[43, 46, 49]', 86: '[48, 51, 55]', 87: '[35, 39, 42, 45]', 88: '[64, 67, 71, 74]', 89: '[65, 69, 72, 76]', 90: '[58, 62, 65]', 91: '[35, 38, 42]', 92: '[39, 42, 45]', 93: '[52, 56, 59]', 94: '[65, 68, 72]', 95: '[47, 51, 54]', 96: '[33, 37, 40, 44]', 97: '[48, 51, 54]', 98: '[58, 61, 65, 68]', 99: '[34, 38, 41, 45]', 100: '[66, 69, 73]', 101: '[48, 52, 55]', 102: '[41, 44, 47]', 103: '[56, 59, 63, 66]', 104: '[44, 48, 51, 55]', 105: '[30, 33, 36]', 106: '[51, 54, 58, 61]', 107: '[40, 43, 47, 50]', 108: '[35, 38, 41]', 109: '[30, 34, 37]', 110: '[42, 46, 49, 52]', 111: '[61, 65, 68, 72]', 112: '[48, 51, 55, 58]', 113: '[59, 62, 65]', 114: '[46, 50, 53, 56]', 115: '[63, 67, 70, 74]', 116: '[57, 60, 64, 67]', 117: '[45, 48, 52, 55]', 118: '[33, 36, 39]', 119: '[51, 54, 58]', 120: '[61, 64, 68, 71]', 121: '[45, 48, 51]', 122: '[50, 54, 57]', 123: '[59, 63, 66, 70]', 124: '[62, 66, 69, 72]', 125: '[68, 72, 75]', 126: '[46, 49, 53, 56]', 127: '[40, 43, 47]', 128: '[49, 53, 56]', 129: '[56, 60, 63, 66]', 130: '[62, 65, 69, 72]', 131: '[69, 73, 76]', 132: '[45, 48, 52]', 133: '[67, 70, 73]', 134: '[52, 56, 59, 63]', 135: '[60, 63, 67]', 136: '[34, 37, 41, 44]', 137: '[34, 38, 41]', 138: '[37, 40, 44]', 139: '[49, 53, 56, 60]', 140: '[33, 37, 40]', 141: '[65, 69, 72]', 142: '[54, 58, 61]', 143: '[50, 54, 57, 60]', 144: '[66, 69, 72]', 145: '[42, 45, 49, 52]', 146: '[41, 45, 48, 51]', 147: '[38, 42, 45, 49]', 148: '[54, 57, 61]', 149: '[64, 67, 71]', 150: '[40, 43, 46]', 151: '[53, 57, 60, 63]', 152: '[60, 64, 67, 70]', 153: '[58, 62, 65, 68]', 154: '[43, 47, 50, 54]', 155: '[34, 37, 41]', 156: '[45, 49, 52]', 157: '[51, 55, 58, 61]', 158: '[56, 59, 63]', 159: '[40, 44, 47, 51]', 160: '[67, 71, 74]', 161: '[63, 66, 69]', 162: '[32, 36, 39]', 163: '[52, 55, 59, 62]', 164: '[45, 49, 52, 56]', 165: '[35, 39, 42]', 166: '[42, 46, 49, 53]', 167: '[60, 64, 67, 71]', 168: '[31, 34, 37]', 169: '[60, 63, 66]', 170: '[57, 60, 63]', 171: '[54, 58, 61, 64]', 172: '[38, 42, 45]', 173: '[50, 54, 57, 61]', 174: '[46, 49, 53]', 175: '[36, 39, 43]', 176: '[32, 35, 39, 42]', 177: '[54, 57, 61, 64]', 178: '[58, 62, 65, 69]', 179: '[47, 50, 53]', 180: '[62, 66, 69]', 181: '[54, 57, 60]', 182: '[31, 35, 38]', 183: '[37, 41, 44, 48]', 184: '[59, 63, 66]', 185: '[53, 56, 60]', 186: '[52, 55, 58]', 187: '[31, 34, 38]', 188: '[47, 51, 54, 58]', 189: '[52, 56, 59, 62]', 190: '[33, 36, 40, 43]', 191: '[56, 60, 63]', 192: '[44, 48, 51, 54]', 193: '[57, 61, 64, 68]', 194: '[49, 53, 56, 59]', 195: '[55, 59, 62, 65]', 196: '[66, 69, 73, 76]', 197: '[62, 66, 69, 73]', 198: '[64, 68, 71]', 199: '[48, 52, 55, 59]', 200: '[42, 45, 49]', 201: '[37, 41, 44, 47]', 202: '[51, 55, 58, 62]', 203: '[39, 43, 46, 50]', 204: '[65, 68, 72, 75]', 205: '[47, 50, 54]', 206: '[64, 68, 71, 75]', 207: '[39, 43, 46, 49]', 208: '[64, 68, 71, 74]', 209: '[58, 61, 65]', 210: '[32, 35, 39]', 211: '[60, 63, 67, 70]', 212: '[54, 58, 61, 65]', 213: '[50, 53, 57]', 214: '[53, 56, 59]', 215: '[40, 44, 47, 50]', 216: '[56, 59, 62]', 217: '[61, 64, 68]', 218: '[41, 45, 48]', 219: '[51, 55, 58]', 220: '[43, 47, 50]', 221: '[62, 65, 68]', 222: '[55, 58, 61]', 223: '[34, 38, 41, 44]'}


def chordoh2interval(chord):
    chord_pitch=CHORD[chord-107]
    chord_pitch=chord_pitch[1:-1].split(',')
    interval=[int(chord_pitch[i + 1]) - int(chord_pitch[i]) for i in range(len(chord_pitch) - 1)]
    return interval


TIMESIGN={0:[6, 8], 1:[4, 4], 2:[9, 8], 3:[2, 4], 4:[3, 4], 5:[2, 2], 6:[6, 4], 7:[3, 2]}
DURATION={0:2, 1:4, 2:6, 3:8, 4:10, 5:12, 6:16, 7:18, 8:20, 9:22, 10:24, 11:30, 12:32, 13:36, 14:42, 15:44, 16:48, 17:54, 18:56, 19:60,
20:64, 21:66, 22:68, 23:72, 24:78, 25:80, 26:84, 27:90, 28:92, 29:96, 30:102, 31:108, 32:120, 33:126, 34:132, 35:138, 36:144}


def getBar(melody):
    barList=[]
    durSum=0
    bar_cnt=0
    for i in range(len(melody)):
        if melody[i]==0:
            if bar_cnt!=0:
                barList.append(durSum)
            durSum=0
            bar_cnt+=1
        if melody[i]>61 and melody[i]<99:
            durSum+=DURATION[melody[i]-62]
    barList.append(durSum)
    return barList


def halfBar_sequences(chord,melody):
    halfBar_chordSeq=[]
    halfBar_melodyPitSeq=[]
    halfBar_melodyDurSeq = []
    j=0;jj=0
    barList=getBar(melody)
    nume=TIMESIGN[melody[0]-99][0]
    deno=TIMESIGN[melody[0]-99][1]
    bar_temp=int(nume*96/deno)
    print("bar: ",len(barList),barList)
    bar=barList[0]
    for i in range(len(melody)):
        if melody[i]>=99:
            continue
        elif melody[i]==0:
            if j!=jj:
                last_pitch=halfBar_melodyPitSeq[-1]
                last_duration=halfBar_melodyDurSeq[-1]
                if len(last_pitch)!=1:
                    print("wrong")
                else:
                    split_pitch=[[last_pitch[0]],[last_pitch[0]]]
                    split_duration=[[last_duration[0]/2],[last_duration[0]/2]]
                    halfBar_melodyPitSeq.pop()
                    halfBar_melodyDurSeq.pop()
                    halfBar_melodyPitSeq.extend(split_pitch)
                    halfBar_melodyDurSeq.extend(split_duration)
                    jj+=1
            halfBar_chordSeq.append(chordRepre2interval(chord[j]))
            halfBar_chordSeq.append(chordRepre2interval(chord[j+1]))
            j+=2
            if j // 2-1 < len(barList):
                bar = barList[j // 2-1]
            melody_pitch = []
            melody_duration = []
            if i+1==len(melody) or melody[i+1]==0:
               for k in DURATION:
                   if DURATION[k]==bar_temp//2:
                       halfBar_melodyPitSeq.append([-1])
                       halfBar_melodyDurSeq.append([bar_temp//2])
                       halfBar_melodyPitSeq.append([-1])
                       halfBar_melodyDurSeq.append([bar_temp//2])
                       jj+=2
        else:
            if melody[i]<=61:
                if melody[i]==1:
                    melody_pitch.append(-1)
                else:
                    melody_pitch.append((melody[i]+40)%12)
            else:
                melody_duration.append(DURATION[melody[i]-62])
                if sum(melody_duration)>=bar/2 or (i<len(melody)-1 and melody[i+1] in [0,99,100,101,102,103,104,105,106])\
                    or (jj%2==0 and sum(melody_duration)<bar/2 and ((i<len(melody)-3 and melody[i+3] in [0,99,100,101,102,103,104,105,106]) or i==len(melody)-3)) \
                    or i==len(melody)-1:
                    if len(melody_pitch)==0 or len(melody_duration)==0:
                        melody_pitch=[1]
                    halfBar_melodyPitSeq.append(melody_pitch)
                    halfBar_melodyDurSeq.append(melody_duration)
                    melody_pitch=[]
                    melody_duration=[]
                    jj+=1
    if j != jj:
        last_pitch = halfBar_melodyPitSeq[-1]
        last_duration = halfBar_melodyDurSeq[-1]
        if len(last_pitch) != 1:
            print("wrong")
        else:
            split_pitch = [[last_pitch[0]], [last_pitch[0]]]
            split_duration = [[last_duration[0] / 2], [last_duration[0] / 2]]
            halfBar_melodyPitSeq.pop()
            halfBar_melodyDurSeq.pop()
            halfBar_melodyPitSeq.extend(split_pitch)
            halfBar_melodyDurSeq.extend(split_duration)
            jj += 1
    print(len(halfBar_chordSeq), halfBar_chordSeq)
    return halfBar_chordSeq,halfBar_melodyPitSeq,halfBar_melodyDurSeq


def compute_metrics(chord_sequence,melody_sequence):
    chord_type={}
    chord_type[0]=0
    for i in range(1,7):
        for j in range(40):
            chord_type[i*100+j]=0
    for i in chord_sequence:
        type=i[:7].index(1)
        root=i[7:].index(1)
        if type!=0 and root!=40:
            chord_type[type*100 + root] += 1
        else:
            chord_type[0] += 1
    ########################## CC & CHE ############################
    chord_statistics=[]
    for i in chord_type:
        chord_statistics.append(chord_type[i])
    CC = len(chord_statistics)-chord_statistics.count(0)
    # # calculate entropy
    chord_statistics = np.array(chord_statistics) / len(chord_statistics)
    CHE = sum([- p_i * np.log(p_i + 1e-6) for p_i in chord_statistics])
    ############################# CTD ############################
    y = 0
    CTD_length=len(chord_sequence) - 1
    for n in range(len(chord_sequence) - 1):
        chord1=chordRepre2interval(chord_sequence[n + 1])
        chord2=chordRepre2interval(chord_sequence[n])
        if len(chord1)>1 and len(chord2)>1:
            y += np.sqrt(
                np.sum((np.asarray(tonal_centroid(chord1))
                        - np.asarray(tonal_centroid(chord2))) ** 2))
        else:
            CTD_length-=1
    if CTD_length==0:
        CTD=1
    else:
        CTD = y / CTD_length
    ################################################################
    chord_zip,melodyPit_zip,melodyDur_zip =halfBar_sequences(chord_sequence,melody_sequence)
    print(len(chord_zip),len(melodyPit_zip),len(melodyDur_zip))
    assert len(chord_zip)==len(melodyPit_zip)==len(melodyDur_zip)
    ############################# CTnCTR ############################
    c = 0
    p = 0
    n = 0
    for melodyPit_m,melodyDur_m,chord_m in zip(melodyPit_zip,melodyDur_zip,chord_zip):
        if len(chord_m)==1:
            for ii in melodyPit_m:
                if ii!=-1:
                    c += 1
            continue
        for i in range(len(melodyPit_m)):
            m = melodyPit_m[i]
            if m != -1:
                if m in chord_m:
                    c += melodyDur_m[i]
                else:
                    n += melodyDur_m[i]
                    if i+1<len(melodyPit_m):
                        if abs(melodyPit_m[i+1] - melodyPit_m[i]) <= 2:
                            p += melodyDur_m[i]
    if (c + n) == 0:
        CTnCTR = 1
    else:
        CTnCTR=(c + p) / (c + n)
    ########################## DC ############################
    DC = 0
    for melodyPit_m, chord_m in zip(melodyPit_zip, chord_zip):
        m_i=0
        for i in range(len(melodyPit_m)):
            m = melodyPit_m[i]
            if m not in chord_m:
                m_i+=1
        if m_i==len(melodyPit_m):
            DC+=1
    DC=DC/len(chord_sequence)
    ############################# PCS ############################
    score = 0
    count = 0
    for melodyPit_m, melodyDur_m, chord_m in zip(melodyPit_zip, melodyDur_zip, chord_zip):
        if len(chord_m)==1:
            continue
        for m in range(len(melodyPit_m)):
            if melodyPit_m[m] != -1:
                score_m=0
                for c in chord_m:
                    # unison, maj, minor 3rd, perfect 5th, maj, minor 6,
                    if (melodyPit_m[m] - c+12)%12 == 0 or (melodyPit_m[m] - c+12)%12 == 3 or (melodyPit_m[m] - c+12)%12 == 4 \
                            or (melodyPit_m[m] - c+12)%12 == 7 or (melodyPit_m[m] - c+12)%12 == 8 or (melodyPit_m[m] - c+12)%12 == 9 \
                            or (melodyPit_m[m] - c+12)%12 == 5:
                        if (melodyPit_m[m] - c+12)%12 ==5:
                            score_m += 0
                        else:
                            score_m += 1
                    else:
                        score_m += -1
                score+=score_m*melodyDur_m[m]
                count+=melodyDur_m[m]
    if count == 0:
        PCS = 1
    else:
        PCS = score / count
    ############################# MCTD ############################
    y = 0
    y_m=0
    count = 0
    for melodyPit_m, melodyDur_m, chord_m in zip(melodyPit_zip, melodyDur_zip, chord_zip):
        if len(chord_m)==1:
            continue
        for m in range(len(melodyPit_m)):
            if melodyPit_m[m] != -1:
                y += np.sqrt(np.sum((np.asarray(tonal_centroid([melodyPit_m[m]])) - np.asarray(tonal_centroid(chord_m)))) ** 2)
                y_m+=y*melodyDur_m[m]
                count += melodyDur_m[m]
    if count == 0:
        MCTD = 1
    else:
        MCTD = y / count
    ############################## CNR #############################
    note_cnt=0
    for i in melody_sequence:
        if i>=1 and i<=98:
            note_cnt+=1
    CNR=len(chord_sequence)/(note_cnt/2)
    return chord_statistics,CNR,DC,CHE,CC,CTD,CTnCTR,PCS,MCTD
