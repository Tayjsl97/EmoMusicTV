import torch
import torch.nn as nn
import os
import pickle
import muspy
import datetime
import random
from chord_metrics import compute_metrics
from models.chordVAE_givenMelody import chordVAE
from utils import calc_chords_val,calc_piece_val

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def computeGS(chordList,melodyList):
    CS_list = []
    sum_CTD = 0;sum_CTnCTR = 0;sum_PCS = 0;sum_MCTD = 0;sum_CNR = 0;sum_DC = 0;sum_CHE = 0;sum_CC = 0
    for i in range(len(chordList)):
        CS,CNR,DC,CHE,CC,CTD,CTnCTR,PCS,MCTD = compute_metrics(chordList[i], melodyList[i])
        CS_list.append(CS)
        sum_CNR += CNR
        sum_DC += DC
        sum_CHE += CHE
        sum_CC += CC
        sum_CTD += CTD
        sum_CTnCTR += CTnCTR
        sum_PCS += PCS
        sum_MCTD += MCTD
    return CS_list,sum_CNR/len(chordList),sum_DC/len(chordList),sum_CHE/len(chordList),sum_CC/len(chordList),\
           sum_CTD/len(chordList),sum_CTnCTR/len(chordList),sum_PCS/len(chordList),sum_MCTD/len(chordList)


def hist_sim(GT_CS,ALL_CS):
    CHS=0
    for i in range(len(ALL_CS)):
        gt=GT_CS[i]
        com=ALL_CS[i]
        similarity=0
        for j in range(len(gt)):
            if gt[j]==0 and com[j]==0:
                similarity += 0
            else:
                similarity+=(1-abs(gt[j]-com[j])/max(gt[j],com[j]))
        similarity /= len(gt)
        # print("similarity: ",similarity)
        CHS+=similarity
    CHS /= len(ALL_CS)
    return CHS


def computeGenVal(chord):
    CHORDTYPE = {0: 'rest', 1: 'm', 2: 'dim', 3: 'maj', 4: 'm7', 5: '7', 6: 'maj7'}
    valence = []
    for j in range(len(chord)):
        if (j + 1) % 2 == 0:
            v1 = CHORDTYPE[chord[j][0:7].index(1)]
            v2 = CHORDTYPE[chord[j - 1][0:7].index(1)]
            valence.append(calc_chords_val([v1, v2]))
    piece_v = calc_piece_val(valence)
    return piece_v,valence


def chordRepre2Pitch(chord):
    CHORDTYPE = {1: [3, 4], 2: [3, 3], 3: [4, 3], 4: [3, 4, 3], 5: [4, 3, 3], 6: [4, 3, 4]}
    chordPitchs=[]
    for i in chord:
        type=i[:7].index(1)
        root=i[7:].index(1)+30
        if type==0 or root==70:
            chordPitchs.append([0])
            continue
        interval=CHORDTYPE[type]
        chordPitch=[root]
        for j in interval:
            chordPitch.append(root+j)
            root+=j
        chordPitchs.append(chordPitch)
    return chordPitchs


CHORD={0: '[0]', 1: '[50, 53, 56]', 2: '[36, 39, 43, 46]', 3: '[67, 70, 74]', 4: '[50, 53, 57, 60]', 5: '[61, 64, 67]', 6: '[32, 36, 39, 42]', 7: '[37, 40, 44, 47]', 8: '[49, 52, 56, 59]', 9: '[55, 59, 62]', 10: '[39, 42, 46, 49]', 11: '[30, 33, 37]', 12: '[41, 44, 48]', 13: '[36, 40, 43]', 14: '[37, 41, 44]', 15: '[47, 50, 54, 57]', 16: '[30, 34, 37, 40]', 17: '[57, 60, 64]', 18: '[49, 52, 56]', 19: '[63, 66, 70]', 20: '[36, 40, 43, 46]', 21: '[55, 58, 62, 65]', 22: '[59, 62, 66]', 23: '[66, 70, 73]', 24: '[55, 58, 62]', 25: '[34, 37, 40]', 26: '[44, 47, 51, 54]', 27: '[38, 41, 45]', 28: '[51, 54, 57]', 29: '[38, 41, 44]', 30: '[62, 65, 69]', 31: '[46, 50, 53, 57]', 32: '[38, 42, 45, 48]', 33: '[43, 47, 50, 53]', 34: '[44, 47, 50]', 35: '[63, 66, 70, 73]', 36: '[46, 49, 52]', 37: '[56, 60, 63, 67]', 38: '[33, 36, 40]', 39: '[44, 48, 51]', 40: '[35, 38, 42, 45]', 41: '[35, 39, 42, 46]', 42: '[37, 40, 43]', 43: '[32, 36, 39, 43]', 44: '[39, 42, 46]', 45: '[42, 45, 48]', 46: '[36, 40, 43, 47]', 47: '[44, 47, 51]', 48: '[58, 61, 64]', 49: '[57, 61, 64]', 50: '[41, 45, 48, 52]', 51: '[59, 62, 66, 69]', 52: '[61, 65, 68]', 53: '[38, 41, 45, 48]', 54: '[68, 71, 75]', 55: '[57, 61, 64, 67]', 56: '[42, 46, 49]', 57: '[33, 37, 40, 43]', 58: '[47, 51, 54, 57]', 59: '[40, 44, 47]', 60: '[36, 39, 42]', 61: '[64, 67, 70]', 62: '[55, 59, 62, 66]', 63: '[53, 56, 60, 63]', 64: '[60, 64, 67]', 65: '[39, 43, 46]', 66: '[63, 67, 70, 73]', 67: '[31, 35, 38, 41]', 68: '[41, 44, 48, 51]', 69: '[59, 63, 66, 69]', 70: '[32, 35, 38]', 71: '[43, 46, 50]', 72: '[46, 50, 53]', 73: '[49, 52, 55]', 74: '[52, 55, 59]', 75: '[45, 49, 52, 55]', 76: '[53, 57, 60, 64]', 77: '[68, 71, 74]', 78: '[65, 68, 71]', 79: '[65, 69, 72, 75]', 80: '[53, 57, 60]', 81: '[48, 52, 55, 58]', 82: '[63, 67, 70]', 83: '[61, 65, 68, 71]', 84: '[43, 46, 50, 53]', 85: '[43, 46, 49]', 86: '[48, 51, 55]', 87: '[35, 39, 42, 45]', 88: '[64, 67, 71, 74]', 89: '[65, 69, 72, 76]', 90: '[58, 62, 65]', 91: '[35, 38, 42]', 92: '[39, 42, 45]', 93: '[52, 56, 59]', 94: '[65, 68, 72]', 95: '[47, 51, 54]', 96: '[33, 37, 40, 44]', 97: '[48, 51, 54]', 98: '[58, 61, 65, 68]', 99: '[34, 38, 41, 45]', 100: '[66, 69, 73]', 101: '[48, 52, 55]', 102: '[41, 44, 47]', 103: '[56, 59, 63, 66]', 104: '[44, 48, 51, 55]', 105: '[30, 33, 36]', 106: '[51, 54, 58, 61]', 107: '[40, 43, 47, 50]', 108: '[35, 38, 41]', 109: '[30, 34, 37]', 110: '[42, 46, 49, 52]', 111: '[61, 65, 68, 72]', 112: '[48, 51, 55, 58]', 113: '[59, 62, 65]', 114: '[46, 50, 53, 56]', 115: '[63, 67, 70, 74]', 116: '[57, 60, 64, 67]', 117: '[45, 48, 52, 55]', 118: '[33, 36, 39]', 119: '[51, 54, 58]', 120: '[61, 64, 68, 71]', 121: '[45, 48, 51]', 122: '[50, 54, 57]', 123: '[59, 63, 66, 70]', 124: '[62, 66, 69, 72]', 125: '[68, 72, 75]', 126: '[46, 49, 53, 56]', 127: '[40, 43, 47]', 128: '[49, 53, 56]', 129: '[56, 60, 63, 66]', 130: '[62, 65, 69, 72]', 131: '[69, 73, 76]', 132: '[45, 48, 52]', 133: '[67, 70, 73]', 134: '[52, 56, 59, 63]', 135: '[60, 63, 67]', 136: '[34, 37, 41, 44]', 137: '[34, 38, 41]', 138: '[37, 40, 44]', 139: '[49, 53, 56, 60]', 140: '[33, 37, 40]', 141: '[65, 69, 72]', 142: '[54, 58, 61]', 143: '[50, 54, 57, 60]', 144: '[66, 69, 72]', 145: '[42, 45, 49, 52]', 146: '[41, 45, 48, 51]', 147: '[38, 42, 45, 49]', 148: '[54, 57, 61]', 149: '[64, 67, 71]', 150: '[40, 43, 46]', 151: '[53, 57, 60, 63]', 152: '[60, 64, 67, 70]', 153: '[58, 62, 65, 68]', 154: '[43, 47, 50, 54]', 155: '[34, 37, 41]', 156: '[45, 49, 52]', 157: '[51, 55, 58, 61]', 158: '[56, 59, 63]', 159: '[40, 44, 47, 51]', 160: '[67, 71, 74]', 161: '[63, 66, 69]', 162: '[32, 36, 39]', 163: '[52, 55, 59, 62]', 164: '[45, 49, 52, 56]', 165: '[35, 39, 42]', 166: '[42, 46, 49, 53]', 167: '[60, 64, 67, 71]', 168: '[31, 34, 37]', 169: '[60, 63, 66]', 170: '[57, 60, 63]', 171: '[54, 58, 61, 64]', 172: '[38, 42, 45]', 173: '[50, 54, 57, 61]', 174: '[46, 49, 53]', 175: '[36, 39, 43]', 176: '[32, 35, 39, 42]', 177: '[54, 57, 61, 64]', 178: '[58, 62, 65, 69]', 179: '[47, 50, 53]', 180: '[62, 66, 69]', 181: '[54, 57, 60]', 182: '[31, 35, 38]', 183: '[37, 41, 44, 48]', 184: '[59, 63, 66]', 185: '[53, 56, 60]', 186: '[52, 55, 58]', 187: '[31, 34, 38]', 188: '[47, 51, 54, 58]', 189: '[52, 56, 59, 62]', 190: '[33, 36, 40, 43]', 191: '[56, 60, 63]', 192: '[44, 48, 51, 54]', 193: '[57, 61, 64, 68]', 194: '[49, 53, 56, 59]', 195: '[55, 59, 62, 65]', 196: '[66, 69, 73, 76]', 197: '[62, 66, 69, 73]', 198: '[64, 68, 71]', 199: '[48, 52, 55, 59]', 200: '[42, 45, 49]', 201: '[37, 41, 44, 47]', 202: '[51, 55, 58, 62]', 203: '[39, 43, 46, 50]', 204: '[65, 68, 72, 75]', 205: '[47, 50, 54]', 206: '[64, 68, 71, 75]', 207: '[39, 43, 46, 49]', 208: '[64, 68, 71, 74]', 209: '[58, 61, 65]', 210: '[32, 35, 39]', 211: '[60, 63, 67, 70]', 212: '[54, 58, 61, 65]', 213: '[50, 53, 57]', 214: '[53, 56, 59]', 215: '[40, 44, 47, 50]', 216: '[56, 59, 62]', 217: '[61, 64, 68]', 218: '[41, 45, 48]', 219: '[51, 55, 58]', 220: '[43, 47, 50]', 221: '[62, 65, 68]', 222: '[55, 58, 61]', 223: '[34, 38, 41, 44]'}


def chordoh2Pitch(chord):
    chordPitchs=[]
    for i in chord:
        chord_pitch = CHORD[i - 107]
        chord_pitch = chord_pitch[1:-1].split(',')
        chord_pitch = [int(k) for k in chord_pitch]
        chordPitchs.append(chord_pitch)
    return chordPitchs


def genefunc(melody,chord,i):
    key_signatures = [muspy.KeySignature(time=0, root=0, mode='major')]
    time_signatures = []
    melody_notes = []
    chord_notes = []
    durAccum = 0
    chord_cnt = 0
    flag = 0
    for j in range(len(melody)):
        if melody[j] >= 99:
            time_signatures.append(muspy.TimeSignature(
                time=durAccum, numerator=TIMESIGN[melody[j] - 99][0],
                denominator=TIMESIGN[melody[j] - 99][1]))
            bar = (96 * TIMESIGN[melody[j] - 99][0]) // TIMESIGN[melody[j] - 99][1]
        elif melody[j] == 0:
            halfDur = 0
            if chord[chord_cnt] == chord[chord_cnt + 1]:
                if len(chord[chord_cnt]) > 1:
                    for chord_pitch in chord[chord_cnt]:
                        chord_notes.append(muspy.Note(time=durAccum, pitch=chord_pitch, duration=bar))
                chord_cnt += 2
                flag = 1
            else:
                flag = 0
        else:
            if melody[j] <= 61:
                if melody[j] == 1:
                    pitch = 0
                else:
                    pitch = melody[j] + 40
            else:
                duration = DURATION[melody[j] - 62]
                if pitch > 0:
                    melody_note = muspy.Note(time=durAccum, pitch=pitch, duration=duration)
                    melody_notes.append(melody_note)
                durAccum += duration
                halfDur += duration
            if flag == 0:
                if halfDur >= (bar // 2) or (j < len(melody) - 1 and melody[j + 1] == 0) or j == len(
                        melody) - 1:
                    if len(chord[chord_cnt]) > 1:
                        for chord_pitch in chord[chord_cnt]:
                            chord_notes.append(muspy.Note(time=durAccum - halfDur, pitch=chord_pitch, duration=halfDur))
                    chord_cnt += 1
                    halfDur = 0
    metadata = muspy.Metadata(schema_version='0.0', source_filename=dataset + "_" + date + "_" + str(i) + '.mid',
                              source_format='midi')
    tempos = [muspy.Tempo(time=0, qpm=120.0)]
    melody_track = muspy.Track(program=0, is_drum=False, name='', notes=melody_notes)
    chord_track = muspy.Track(program=0, is_drum=False, name='', notes=chord_notes)
    music_track = []
    music_track.append(melody_track);
    music_track.append(chord_track)
    music = muspy.Music(metadata=metadata, resolution=24, tempos=tempos,
                        key_signatures=key_signatures, time_signatures=time_signatures, tracks=music_track)
    music_disPath = os.path.join(disPath, "GT_chordVAE_" + dataset + "_" + date + "_" + str(i) + '.mid')
    muspy.write_midi(music_disPath, music)

TIMESIGN={0:[6, 8], 1:[4, 4], 2:[9, 8], 3:[2, 4], 4:[3, 4], 5:[2, 2], 6:[6, 4], 7:[3, 2]}
DURATION={0:2, 1:4, 2:6, 3:8, 4:10, 5:12, 6:16, 7:18, 8:20, 9:22, 10:24, 11:30, 12:32, 13:36, 14:42, 15:44, 16:48, 17:54, 18:56, 19:60,
20:64, 21:66, 22:68, 23:72, 24:78, 25:80, 26:84, 27:90, 28:92, 29:96, 30:102, 31:108, 32:120, 33:126, 34:132, 35:138, 36:144}
@torch.no_grad()
def generate(model,melodyList,chordList,valenceList,disPath,generate_num):
    if not os.path.exists(disPath):
        os.makedirs(disPath)
    GT_CS, GT_CNR,GT_DC,GT_CHE,GT_CC, GT_CTD, GT_CTnCTR, GT_PCS, GT_MCTD = computeGS(chordList[0:generate_num],melodyList[0:generate_num])
    # for i in range(generate_num):
    #     if i%6==0:
    #         chord=chordRepre2Pitch(chordList[i])
    #         genefunc(melodyList[i],chord,i)
    # return
    all_CS=[]; sum_CTD = 0;sum_CTnCTR = 0;sum_PCS = 0;sum_MCTD = 0;sum_CNR = 0;sum_DC=0;sum_CHE=0;sum_CC=0
    genP_V = [];realP_V=[];barAcc = 0
    for i in range(generate_num):
        print("-------------------------------"+str(i)+"--------------------------------")
        melody=torch.LongTensor(melodyList[i]).unsqueeze(0).to(device)
        valence=valenceList[i]
        s_p=torch.nn.functional.one_hot(torch.LongTensor([valence[0]])+2,num_classes=5).unsqueeze(0).float().to(device)
        S_B=torch.nn.functional.one_hot(torch.LongTensor(valence[1:])+2,num_classes=5).unsqueeze(0).float().to(device)
        chord=model.generate(melody,s_p,S_B)
        chord=chord.squeeze(0).cpu().tolist()
        # print(chord)
        ######################## compute metrics ######################################
        if metrics is True:
            CS,CNR,DC,CHE,CC,CTD,CTnCTR,PCS,MCTD = compute_metrics(chord,melodyList[i])
        else:
            temp=[0]*241
            CS, CNR, DC, CHE, CC, CTD, CTnCTR, PCS, MCTD=temp,0,0,0,0,0,0,0,0
        all_CS.append(CS)
        sum_CNR += CNR;sum_DC += DC;sum_CHE += CHE;sum_CC += CC
        sum_CTD += CTD;sum_CTnCTR += CTnCTR;sum_PCS += PCS;sum_MCTD += MCTD
        ######################### compute sentiment ####################################
        WholeVal,ValSeq=computeGenVal(chord)
        genP_V.append(WholeVal)
        realP_V.append(valence[0])
        print("real_valence: ", valence[0], valence[1:])
        print("gen_valence: ", WholeVal, ValSeq)
        accB = 0
        for k in range(len(ValSeq)):
            if ValSeq[k] == valence[1:][k]:
                accB += 1
        barAcc += (accB / len(ValSeq))
        ########################## generate MIDI ###################################
        chord=chordRepre2Pitch(chord)
        if GENERATE and i%6==0:
            genefunc(melodyList[i],chord,i)
    CHS = hist_sim(GT_CS, all_CS)
    cnt = 0
    for i in range(len(realP_V)):
        if genP_V[i] == realP_V[i]:
            cnt += 1
    return (1, round(CHS, 4)), (round(GT_CNR, 4), round(sum_CNR / generate_num, 4)), \
           (round(GT_DC, 4), round(sum_DC / generate_num, 4)), \
           (round(GT_CHE, 4), round(sum_CHE / generate_num, 4)), \
           (round(GT_CC, 4), round(sum_CC / generate_num, 4)), \
           (round(GT_CTD, 4), round(sum_CTD / generate_num, 4)), \
           (round(GT_CTnCTR, 4), round(sum_CTnCTR / generate_num, 4)), \
           (round(GT_PCS, 4), round(sum_PCS / generate_num, 4)), \
           (round(GT_MCTD, 4), round(sum_MCTD / generate_num, 4)), \
           round(cnt / len(realP_V), 4),round(barAcc/generate_num,4)


if __name__ == '__main__':

    dataset = "NMD"
    date = str(datetime.date.today())
    GENERATE = False
    metrics = True
    generate_num = 1000
    disPath='./generated_music/chord/'
    # load models
    resume="./save_models/chordVAE_NMD_epoch96_min_0.0832.pth"
    model=chordVAE(N=3,h=4,m_size=8,c_size=48,d_ff=256,hidden_size=256,latent_size=128,dropout=0.2).to(device)
    dict=torch.load(resume,map_location=device)
    model.load_state_dict(dict['model'])
    model.eval()

    file = open("./data/" + dataset + "_melody_test.data", 'rb')
    test_melody = pickle.load(file)
    file = open("./data/" + dataset + "_chord_test.data", 'rb')
    test_chord = pickle.load(file)
    file = open("./data/" + dataset + "_chord_test_valence.data", 'rb')
    test_valence = pickle.load(file)
    print(len(test_melody), len(test_chord), len(test_valence))

    CHS,CNR,DC,CHE,CC,CTD,CTnCTR,PCS,MCTD,acc,accBar=generate(model,test_melody,test_chord,test_valence,disPath,generate_num)
    print(CHS, CNR, DC, CHE, CC, CTD, CTnCTR, PCS, MCTD)
    print(acc, accBar)

