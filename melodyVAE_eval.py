import torch
import os
import pickle
import muspy
import datetime
from models.melodyVAE_givenHarmony import melodyVAE
from melody_metrics import compute_metrics

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def computeGS(chordList,melodyList):
    sum_PCHE = 0;sum_DCHE = 0;sum_API = 0;sum_HC = 0;sum_CTnCTR = 0;sum_PCS = 0;sum_MCTD = 0
    for i in range(len(melodyList)):
        PCHE,DCHE,API,HC,CTnCTR,PCS,MCTD = compute_metrics(chordList[i], melodyList[i])
        sum_PCHE += PCHE
        sum_DCHE += DCHE
        sum_API += API
        sum_HC += HC
        sum_CTnCTR += CTnCTR
        sum_PCS += PCS
        sum_MCTD += MCTD
        print(sum_PCHE,sum_DCHE)
    return sum_PCHE/len(melodyList),sum_DCHE/len(melodyList),sum_API/len(melodyList),sum_HC/len(melodyList),\
           sum_CTnCTR/len(melodyList),sum_PCS/len(melodyList),sum_MCTD/len(melodyList)


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


def genefunc(melody,chord,i):
    key_signatures = [muspy.KeySignature(time=0, root=0, mode='major')]
    time_signatures = []
    melody_notes = []
    chord_notes = []
    durAccum = 0
    barDur = 0
    chord_cnt = 0
    for j in range(len(melody)):
        if melody[j] >= 99:
            time_signatures.append(muspy.TimeSignature(
                time=durAccum, numerator=TIMESIGN[melody[j] - 99][0],
                denominator=TIMESIGN[melody[j] - 99][1]))
            bar = (96 * TIMESIGN[melody[j] - 99][0]) // TIMESIGN[melody[j] - 99][1]
            continue
        if melody[j] == 0:
            if chord[chord_cnt] == chord[chord_cnt + 1]:
                if len(chord[chord_cnt]) > 1:
                    for chord_pitch in chord[chord_cnt]:
                        chord_notes.append(muspy.Note(time=(chord_cnt // 2) * bar, pitch=chord_pitch, duration=bar))
            else:
                if len(chord[chord_cnt]) > 1:
                    for chord_pitch in chord[chord_cnt]:
                        chord_notes.append(
                            muspy.Note(time=(chord_cnt // 2) * bar, pitch=chord_pitch, duration=bar // 2))
                if len(chord[chord_cnt + 1]) > 1:
                    for chord_pitch in chord[chord_cnt+1]:
                        chord_notes.append(
                            muspy.Note(time=(chord_cnt // 2) * bar + bar // 2, pitch=chord_pitch, duration=bar // 2))
            chord_cnt += 2
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
    metadata = muspy.Metadata(schema_version='0.0', source_filename=dataset + "_" + date + "_" + str(i) + '.mid',
                              source_format='midi')
    tempos = [muspy.Tempo(time=0, qpm=110.0)]
    melody_track = muspy.Track(program=0, is_drum=False, name='', notes=melody_notes)
    chord_track = muspy.Track(program=0, is_drum=False, name='', notes=chord_notes)
    music_track = []
    music_track.append(melody_track);
    music_track.append(chord_track)
    music = muspy.Music(metadata=metadata, resolution=24, tempos=tempos,
                        key_signatures=key_signatures, time_signatures=time_signatures, tracks=music_track)
    music_disPath = os.path.join(disPath, "GT_melodyVAE_" + dataset + "_" + date + "_" + str(i) + '.mid')
    muspy.write_midi(music_disPath, music)


TIMESIGN={0:[6, 8], 1:[4, 4], 2:[9, 8], 3:[2, 4], 4:[3, 4], 5:[2, 2], 6:[6, 4], 7:[3, 2]}
DURATION={0:2, 1:4, 2:6, 3:8, 4:10, 5:12, 6:16, 7:18, 8:20, 9:22, 10:24, 11:30, 12:32, 13:36, 14:42, 15:44, 16:48, 17:54, 18:56, 19:60,
20:64, 21:66, 22:68, 23:72, 24:78, 25:80, 26:84, 27:90, 28:92, 29:96, 30:102, 31:108, 32:120, 33:126, 34:132, 35:138, 36:144}
@torch.no_grad()
def generate(model,melodyList,chordList,valenceList,disPath,generate_num):
    if not os.path.exists(disPath):
        os.makedirs(disPath)
    GT_PCHE,GT_DCHE,GT_API,GT_HC,GT_CTnCTR,GT_PCS,GT_MCTD = computeGS(chordList[0:generate_num],melodyList[0:generate_num])
    # for i in range(generate_num):
    #     if i%6==0:
    #         chord=chordRepre2Pitch(chordList[i])
    #         genefunc(melodyList[i],chord,i)
    # return
    print(GT_PCHE,GT_DCHE,GT_API,GT_HC,GT_CTnCTR,GT_PCS,GT_MCTD)
    sum_PCHE = 0;sum_DCHE = 0;sum_API = 0;sum_HC = 0;sum_CTnCTR = 0;sum_PCS = 0;sum_MCTD = 0
    wrong_cnt=0
    for i in range(generate_num):
        print("-------------------------------"+str(i)+"--------------------------------")
        valence=valenceList[i]
        chord=torch.Tensor(chordList[i]).unsqueeze(0).to(device)
        s_p=torch.nn.functional.one_hot(torch.LongTensor([valence[0]])+2,num_classes=5).unsqueeze(0).float().to(device)
        S_B=torch.nn.functional.one_hot(torch.LongTensor(valence[1:])+2,num_classes=5).unsqueeze(0).float().to(device)
        timeSign=[0]*8
        timeSign[melodyList[i][0]-99]=1
        timeSign=torch.tensor(timeSign).unsqueeze(0).float().to(device)
        melody=model.old_gen(chord,s_p,S_B,timeSign)
        if melody==0:
            wrong_cnt+=1
            continue
        melody.insert(0,melodyList[i][0]);melody.insert(1,0);melody.pop()
        print(len(melody),melody)
        ######################## compute metrics ######################################
        if metrics is True:
            PCHE,DCHE,API,HC,CTnCTR,PCS,MCTD = compute_metrics(chordList[i],melody)
        else:
            PCHE,DCHE,API,HC,CTnCTR,PCS,MCTD=0,0,0,0,0,0,0
        sum_PCHE += PCHE;sum_DCHE += DCHE;sum_API += API;sum_HC += HC
        sum_CTnCTR += CTnCTR;sum_PCS += PCS;sum_MCTD += MCTD
        ########################## generate MIDI ###################################
        barList = getBar(melody)
        chord=chordRepre2Pitch(chordList[i])
        if gen and i%6==0:
            genefunc(melody,chord,i)
    return (round(GT_PCHE, 4), round(sum_PCHE / (generate_num-wrong_cnt), 4)), \
           (round(GT_DCHE, 4), round(sum_DCHE / (generate_num-wrong_cnt), 4)), \
           (round(GT_API, 4), round(sum_API / (generate_num-wrong_cnt), 4)), \
           (round(GT_HC, 4), round(sum_HC / (generate_num-wrong_cnt), 4)), \
           (round(GT_CTnCTR, 4), round(sum_CTnCTR / (generate_num-wrong_cnt), 4)), \
           (round(GT_PCS, 4), round(sum_PCS / (generate_num-wrong_cnt), 4)), \
           (round(GT_MCTD, 4), round(sum_MCTD / (generate_num-wrong_cnt), 4))


if __name__ == '__main__':

    dataset = "NMD"
    date = str(datetime.date.today())
    gen = False
    metrics = True
    generate_num = 1000
    disPath='./generated_music/melody/'
    # load models
    resume="./save_models/melodyVAE_NMD_epoch104_min_0.5064.pth"
    model = melodyVAE(N=3,h=4,m_size=8,c_size=48,d_ff=256,hidden_size=256,latent_size=128,dropout=0.2).to(device)
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

    PCHE,DCHE,API,HC,CTnCTR,PCS,MCTD=generate(model,test_melody,test_chord,test_valence,disPath,generate_num)
    print(PCHE,DCHE,API,HC,CTnCTR,PCS,MCTD)
