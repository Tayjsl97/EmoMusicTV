"""
Created on Fri Oct 29 17:28:18 2022
@author: Shulei Ji
"""

import torch
import os
import pickle
import muspy
from models.EmoMusicTV import EmoMusicTV
from leadsheet_metrics import compute_metrics
from chord_metrics import getBar
from utils import calc_chords_val,calc_piece_val

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def computeGS(chordList,melodyList):
    sum_PCHE = 0;sum_DCHE = 0;sum_API = 0;sum_HC = 0;
    sum_CC = 0;sum_CHE = 0;sum_CTD = 0;sum_DC = 0;
    sum_CTnCTR = 0;sum_PCS = 0;sum_MCTD = 0
    for i in range(len(melodyList)):
        PCHE,DCHE,API,HC,CC,CHE,CTD,DC,CTnCTR,PCS,MCTD = compute_metrics(chordList[i], melodyList[i])
        sum_PCHE += PCHE;sum_DCHE += DCHE;sum_API += API;sum_HC += HC;
        sum_CC+=CC;sum_CHE+=CHE;sum_CTD+=CTD;sum_DC+=DC;
        sum_CTnCTR += CTnCTR;sum_PCS += PCS;sum_MCTD += MCTD
    return sum_PCHE/len(melodyList),sum_DCHE/len(melodyList),sum_API/len(melodyList),sum_HC/len(melodyList),\
           sum_CC/len(melodyList),sum_CHE/len(melodyList),sum_CTD/len(melodyList),sum_DC/len(melodyList),\
           sum_CTnCTR/len(melodyList),sum_PCS/len(melodyList),sum_MCTD/len(melodyList)


def computeGenVal(chord,all_chord):
    CHORDTYPE = {0: 'rest', 1: 'm', 2: 'dim', 3: 'maj', 4: 'm7', 5: '7', 6: 'maj7'}
    valence = []
    for j in range(len(chord)):
        if (j + 1) % 2 == 0:
            v1 = CHORDTYPE[chord[j][0:7].index(1)]
            v2 = CHORDTYPE[chord[j - 1][0:7].index(1)]
            valence.append(calc_chords_val([v1, v2]))
    valence_temp=[]
    for j in range(len(all_chord)):
        if (j + 1) % 2 == 0:
            v1 = CHORDTYPE[all_chord[j][0:7].index(1)]
            v2 = CHORDTYPE[all_chord[j - 1][0:7].index(1)]
            valence_temp.append(calc_chords_val([v1, v2]))
    piece_v = calc_piece_val(valence_temp)
    return piece_v,valence


def chordRepre2Pitch(chord):
    CHORDTYPE = {1: [3, 4], 2: [3, 3], 3: [4, 3], 4: [3, 4, 3], 5: [4, 3, 3], 6: [4, 3, 4]}
    chordPitchs=[]
    for i in chord:
        type=i[:7].index(1)
        root=i[7:].index(1)+30
        if type==0:
            chordPitchs.append([0])
            continue
        interval=CHORDTYPE[type]
        chordPitch=[root]
        for j in interval:
            chordPitch.append(root+j)
            root+=j
        chordPitchs.append(chordPitch)
    return chordPitchs


def get_bars_Melody(bar_len,melody,i,last_timesign):
        bar_cnt=0
        piece_start=1
        for j in range(len(melody)):
            if melody[j]>=99:
                last_timesign=melody[j]
            elif melody[j]==0 or j==len(melody)-1:
                if melody[j]==0 and j==len(melody)-1:
                    melody_temp=[0]
                    return last_timesign,melody_temp
                bar_cnt+=1
                if j==len(melody)-1:
                   j+=1
                if bar_cnt!=1 and (bar_cnt-1)%bar_len==0:
                    if (bar_cnt-1)//bar_len==(i+1):
                        melody_temp=melody[piece_start:j]
                        if melody_temp[-1]>=99:
                            melody_temp.pop()
                        return last_timesign,melody_temp
                    else:
                        piece_start=j


def generate_GT(melody,chord,i):
    chord = chordRepre2Pitch(chord)
    barList = getBar(melody)
    key_signatures = [muspy.KeySignature(time=0, root=0, mode='major')]
    time_signatures = []
    melody_notes = []
    chord_notes = []
    durAccum = 0
    barDur = 0
    chord_cnt = 0
    for j in range(len(melody)):
        if melody[j] >= 99:
            time_signatures.append(muspy.TimeSignature(time=durAccum, numerator=TIMESIGN[melody[j] - 99][0],
                                                       denominator=TIMESIGN[melody[j] - 99][1]))
            bar = int(TIMESIGN[melody[j] - 99][0] * 96 / TIMESIGN[melody[j] - 99][1])
            continue
        if melody[j] == 0:
            flag = 0
            print(chord_cnt // 2)
            bar = barList[chord_cnt // 2]
            if chord[chord_cnt] == chord[chord_cnt + 1]:
                flag = 1
                if len(chord[chord_cnt]) > 1:
                    for chord_pitch in chord[chord_cnt]:
                        chord_notes.append(muspy.Note(time=durAccum, pitch=chord_pitch, duration=bar))
                chord_cnt += 2
            barDur = 0
            jj = 0
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
                barDur += duration
                if flag == 0:
                    if barDur >= bar / 2 or (j < len(melody) - 1 and melody[j + 1] in [0, 99, 100, 101, 102, 103, 104, 105, 106]) \
                            or (jj % 2 == 0 and barDur < bar / 2 and ((j < len(melody) - 2 and melody[j + 2] in [0, 99, 100, 101, 102, 103, 104, 105, 106])
                                                                      or j == len(melody) - 2)) or j == len(melody) - 1:
                        if len(chord[chord_cnt]) > 1:
                            for chord_pitch in chord[chord_cnt]:
                                chord_notes.append(
                                    muspy.Note(time=durAccum - barDur, pitch=chord_pitch, duration=barDur))
                        chord_cnt += 1
                        jj += 1
                        barDur = 0
    metadata = muspy.Metadata(schema_version='0.0',
                              source_filename="GT_" + str(i) + '.mid',
                              source_format='midi')
    tempos = [muspy.Tempo(time=0, qpm=120.0)]
    melody_track = muspy.Track(program=0, is_drum=False, name='', notes=melody_notes)
    chord_track = muspy.Track(program=0, is_drum=False, name='', notes=chord_notes)
    music_track = []
    music_track.append(melody_track);
    music_track.append(chord_track)
    music = muspy.Music(metadata=metadata, resolution=24, tempos=tempos,
                        key_signatures=key_signatures, time_signatures=time_signatures, tracks=music_track)
    music_disPath = os.path.join(disPath, "inpainting_" + str(i) + '.mid')
    muspy.write_midi(music_disPath, music)

import copy
TIMESIGN={0:[6, 8], 1:[4, 4], 2:[9, 8], 3:[2, 4], 4:[3, 4], 5:[2, 2], 6:[6, 4], 7:[3, 2]}
TIMEDICT={72:[6,8],96:[4,4],108:[9,8],48:[2,4],144:[6,4],60:[5,8],84:[7,8],120:[5,4],24:[1,4]}
DURATION={0:2, 1:4, 2:6, 3:8, 4:10, 5:12, 6:16, 7:18, 8:20, 9:22, 10:24, 11:30, 12:32, 13:36, 14:42, 15:44, 16:48, 17:54, 18:56, 19:60,
20:64, 21:66, 22:68, 23:72, 24:78, 25:80, 26:84, 27:90, 28:92, 29:96, 30:102, 31:108, 32:120, 33:126, 34:132, 35:138, 36:144}
def generate(model,melodyList,chordList,valenceList,disPath,generate_num):
    if not os.path.exists(disPath):
        os.makedirs(disPath)
    melody_pre = [i[0] for i in melodyList]
    melody_post = [i[1] for i in melodyList]
    melodyTemp=copy.deepcopy(melodyList)
    melody_all=[]
    for i in range(len(melodyTemp)):
       melodyTemp[i][0].extend(melodyTemp[i][1])
       melody_all.append(melodyTemp[i][0])
    GT_PCHE, GT_DCHE, GT_API, GT_HC, GT_CC, GT_CHE, GT_CTD, GT_DC, GT_CTnCTR, GT_PCS, GT_MCTD = \
            computeGS([k[24:] for k in chordList[:generate_num]], melody_post[:generate_num])
    if GT==True:
        for i in range(generate_num):
            if i%1==0:
                generate_GT(melody_all[i],chordList[i],i)
    print(GT_PCHE, GT_DCHE, GT_API, GT_HC, GT_CC, GT_CHE, GT_CTD, GT_DC, GT_CTnCTR, GT_PCS, GT_MCTD)
    sum_PCHE = 0;sum_DCHE = 0;sum_API = 0;sum_HC = 0;
    sum_CC = 0;sum_CHE = 0;sum_CTD = 0;sum_DC = 0;
    sum_CTnCTR = 0;sum_PCS = 0;sum_MCTD = 0
    genP_V = [];realP_V = [];barAcc = 0
    for i in range(0,generate_num):
        print("-------------------------------"+str(i)+"--------------------------------")
        valence=valenceList[i]
        chord_pre_one=torch.Tensor(chordList[i][:24]).unsqueeze(0).to(device)
        melody_pre_one=torch.Tensor(melody_pre[i]).unsqueeze(0).long().to(device)
        s_p=torch.nn.functional.one_hot(torch.LongTensor([valence[0]])+2,num_classes=5).unsqueeze(0).float().to(device)
        S_B=torch.nn.functional.one_hot(torch.LongTensor(valence[13:])+2,num_classes=5).unsqueeze(0).float().to(device)
        timeSign=torch.tensor(melody_post[i][0]).unsqueeze(0).long().to(device)
        timeSign=torch.nn.functional.one_hot(timeSign-99,num_classes=8).float()
        melody,chord=model.generate(melody_pre_one,chord_pre_one,s_p,S_B,timeSign)
        melody.insert(0, int(melody_post[i][0]));melody.insert(1,0);melody.pop()
        all_melody=melody_pre[i]
        all_melody.extend(melody)
        all_chord=chordList[i][:24]
        all_chord.extend(chord)
        ######################## metrics ######################################
        if metrics is True:
            PCHE, DCHE, API, HC, CC, CHE, CTD, DC, CTnCTR, PCS, MCTD = compute_metrics(chord, melody)
        else:
            PCHE, DCHE, API, HC, CC, CHE, CTD, DC, CTnCTR, PCS, MCTD = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        sum_PCHE += PCHE;sum_DCHE += DCHE;sum_API += API;sum_HC += HC
        sum_CC += CC;sum_CHE += CHE;sum_CTD += CTD;sum_DC += DC
        sum_CTnCTR += CTnCTR;sum_PCS += PCS;sum_MCTD += MCTD
        ######################### sentiment ####################################
        WholeVal, ValSeq = computeGenVal(chord,all_chord)
        genP_V.append(WholeVal)
        realP_V.append(valence[0])
        print("real_valence: ", valence[0], valence[13:])
        print("gen_valence: ", WholeVal, ValSeq)
        accB = 0
        for k in range(len(ValSeq)):
            if ValSeq[k] == valence[13:][k]:
                accB += 1
        barAcc += (accB / len(ValSeq))
        ########################## generate MIDI #################################
        chord = chordRepre2Pitch(all_chord)
        melody=all_melody
        barList=getBar(melody)
        print(len(chord),melody.count(0),len(barList))
        if GEN and i % 1==0:
            key_signatures = [muspy.KeySignature(time=0, root=0, mode='major')]
            nume = TIMESIGN[melodyList[i][1][0] - 99][0]
            deno = TIMESIGN[melodyList[i][1][0] - 99][1]
            time_signatures = []
            melody_notes = []
            chord_notes = []
            bar = int(nume * 96 / deno)
            durAccum = 0
            chordDur = 0
            barDur = 0
            chord_cnt = 0
            for j in range(len(melody)):
                if melody[j] >= 99:
                    time_signatures.append(muspy.TimeSignature(time=durAccum, numerator=TIMESIGN[melody[j] - 99][0],
                                                               denominator=TIMESIGN[melody[j] - 99][1]))
                    bar = int(TIMESIGN[melody[j] - 99][0] * 96 / TIMESIGN[melody[j] - 99][1])
                    continue
                if melody[j] == 0:
                    if chord[chord_cnt] == chord[chord_cnt + 1]:
                        if len(chord[chord_cnt]) > 1:
                            for chord_pitch in chord[chord_cnt]:
                                chord_notes.append(muspy.Note(time=chordDur, pitch=chord_pitch, duration=bar))
                    else:
                        if len(chord[chord_cnt]) > 1:
                            for chord_pitch in chord[chord_cnt]:
                                chord_notes.append(
                                    muspy.Note(time=chordDur, pitch=chord_pitch, duration=bar // 2))
                        if len(chord[chord_cnt + 1]) > 1:
                            for chord_pitch in chord[chord_cnt+1]:
                                chord_notes.append(
                                    muspy.Note(time=chordDur + bar // 2, pitch=chord_pitch, duration=bar // 2))
                    chord_cnt += 2
                    chordDur += bar
                    barDur = 0
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
                        barDur += duration
            metadata = muspy.Metadata(schema_version='0.0',
                                      source_filename=str(i) + '.mid',
                                      source_format='midi')
            tempos = [muspy.Tempo(time=0, qpm=120.0)]
            melody_track = muspy.Track(program=0, is_drum=False, name='', notes=melody_notes)
            chord_track = muspy.Track(program=0, is_drum=False, name='', notes=chord_notes)
            music_track = []
            music_track.append(melody_track);
            music_track.append(chord_track)
            music = muspy.Music(metadata=metadata, resolution=24, tempos=tempos,
                                key_signatures=key_signatures, time_signatures=time_signatures, tracks=music_track)
            music_disPath = os.path.join(disPath, "Inpainting_412_" + str(i) + '.mid')
            muspy.write_midi(music_disPath, music)
    cnt = 0
    for i in range(len(realP_V)):
        if genP_V[i] == realP_V[i]:
            cnt += 1
    return (round(GT_PCHE, 4), round(sum_PCHE / generate_num, 4)), \
           (round(GT_DCHE, 4), round(sum_DCHE / generate_num, 4)), \
           (round(GT_API, 4), round(sum_API / generate_num, 4)), \
           (round(GT_HC, 4), round(sum_HC / generate_num, 4)), \
           (round(GT_CC, 4), round(sum_CC / generate_num, 4)), \
           (round(GT_CHE, 4), round(sum_CHE / generate_num, 4)), \
           (round(GT_CTD, 4), round(sum_CTD / generate_num, 4)), \
           (round(GT_DC, 4), round(sum_DC / generate_num, 4)), \
           (round(GT_CTnCTR, 4), round(sum_CTnCTR / generate_num, 4)), \
           (round(GT_PCS, 4), round(sum_PCS / generate_num, 4)), \
           (round(GT_MCTD, 4), round(sum_MCTD / generate_num, 4)), \
           round(cnt / len(realP_V), 4), round(barAcc / generate_num, 4)


if __name__ == '__main__':
    GEN = True
    metrics = True
    GT=True
    generate_num = 1000
    disPath='./generated_music/'
    # load models
    resume="./save_models/your_pretrained_models.pth"
    model = EmoMusicTV(N=3,h=4,m_size=8,c_size=48,d_ff=256,hidden_size=256,latent_size=128,dropout=0.2).to(device)
    dict=torch.load(resume,map_location=device)
    model.load_state_dict(dict['model'])
    model.eval()

    file = open("./data/All_124_melody_test.data", 'rb')
    test_melody = pickle.load(file)
    file = open("./data/All_124_chord_test.data", 'rb')
    test_chord = pickle.load(file)
    file = open("./data/All_124_valence_test.data", 'rb')
    test_valence = pickle.load(file)
    print(len(test_melody), len(test_chord), len(test_valence))

    PCHE, DCHE, API, HC, CC, CHE, CTD, DC, CTnCTR, PCS, MCTD,acc,accBar=generate(model,test_melody[:],test_chord[:],test_valence[:],disPath,generate_num)
    print(PCHE, DCHE, API, HC, CC, CHE, CTD, DC, CTnCTR, PCS, MCTD)
    print(acc, accBar)