"""
Created on Mon Nov 20 20:03:00 2023
@author: Shulei Ji
"""
import os
from music21 import *
import muspy
import shutil
import pickle
import numpy as np
from LS_train_utils import *


def delete_mono_all(path):
    """
        delete monophonic music
    """
    midi_file=os.listdir(path)
    for midi_file_one in midi_file:
        midi_path=os.path.join(path,midi_file_one)
        streamm = converter.parse(midi_path)
        if len(streamm)!=2:
            os.remove(midi_path)
            break
        else:
            melodys = streamm[0]
            chords = streamm[1]
        parts = instrument.partitionByInstrument(streamm)
        if len(parts.parts)!=1:
            print("parts: ",len(parts.parts))
        melody_parts=instrument.partitionByInstrument(melodys)
        chord_parts = instrument.partitionByInstrument(chords)
        if parts is None or melody_parts is None or chord_parts is None:
            print("None: ", midi_path)
            os.remove(midi_path)
            break
        notes = parts.parts[0].recurse()
        note_cnt = 0
        chord_cnt = 0
        for i in notes:
            if isinstance(i, note.Note):
                note_cnt += 1
            if isinstance(i, chord.Chord):
                chord_cnt += 1
        melody_notes=melody_parts.parts[0].recurse()
        melody_flag=0
        for i in melody_notes:
            if isinstance(i, chord.Chord):
                print("wrong1: ", midi_path)
                melody_flag=1
                break
        chord_notes = chord_parts.parts[0].recurse()
        chord_flag = 0
        for i in chord_notes:
            if isinstance(i, note.Note):
                print("wrong2: ", midi_path)
                chord_flag=1
                break
        if note_cnt==0 or chord_cnt==0 or melody_flag==1 or chord_flag==1:
            print("del file: ",midi_path)
            os.remove(midi_path)


def delTie(chords):
    """
        delete tie on the chords
    """
    newChords=[]
    dur_sum = 0
    bar_num = 0
    for c_note in chords:
        if isinstance(c_note, meter.TimeSignature):
            bar = (96 * c_note.numerator) // c_note.denominator
            dur_sum = 0
            newChords.append(c_note)
        if isinstance(c_note, note.Rest) or isinstance(c_note, chord.Chord):
            if dur_sum % bar == 0:
                bar_num += 1
            note_duration = int(c_note.duration.quarterLength * 24)
            if isinstance(c_note, chord.Chord):
                note_pitch = [i.pitch.midi for i in list(c_note)]
            else:
                note_pitch = [0]
            if ((dur_sum + note_duration) // bar + 1) > bar_num:
                dur1 = bar - (dur_sum % bar)
                if dur1!=bar:
                    newChords.append([note_pitch, dur1])
                    bar_num+=1
                else:
                    dur1=0
                for k in range((note_duration - dur1) // bar):
                    newChords.append([note_pitch, bar])
                    bar_num+=1
                if (note_duration - dur1) % bar != 0:
                    newChords.append([note_pitch, note_duration - dur1 - ((note_duration - dur1) // bar) * bar])
                else:
                    bar_num-=1
            else:
                newChords.append([note_pitch, note_duration])
                if (dur_sum + note_duration) % bar == 0 and (dur_sum + note_duration) // bar>bar_num:
                    bar_num = ((dur_sum + note_duration) // bar -1)
            dur_sum += note_duration
    return newChords


def pitch_range(midi_path):
    music = muspy.read_midi(midi_path, 'pretty_midi')
    pitch_max = 0;pitch_min = 128
    melody=music.tracks[0].notes
    pitches = np.array([note.pitch for note in melody if note.end > 0])
    maxx = np.max(pitches)
    minn = np.min(pitches)
    if maxx>pitch_max:
        pitch_max=maxx
    if minn<pitch_min:
        pitch_min=minn
    return pitch_min,pitch_max


def root_octave(midi_path):
    stream = converter.parse(midi_path)
    parts = instrument.partitionByInstrument(stream)
    notes = parts.parts[0].recurse()
    root = set()
    for i in notes:
        if isinstance(i, chord.Chord):
            root.add(i.pitches[0].octave)
    return root


def transpose_midi(path,min_pitch=48,max_pitch=95):
    """
        Transpose music
        Ensuring that the pitch range stays within [min_pitch, max_pitch]
    """
    midi_files = os.listdir(path)
    error = set()
    for midi_file in midi_files:
        midi_path = os.path.join(path, midi_file)
        score = muspy.read_midi(midi_path, 'pretty_midi')
        # key=score.key_signatures
        index = midi_path.rfind("\\")
        path = midi_path[:index]
        name = midi_path[index + 1:]
        path = path.replace("train", "train_trans")
        path = path.replace("test", "test_trans")
        name = name.replace(".mid", "")
        if not os.path.exists(path):
            os.makedirs(path)
        midi_path = os.path.join(path, name)
        for i in range(-6, 6, 1):
            flag = 0
            newscore = score.transpose(i)
            # newkey = newscore.key_signatures
            midi_path_new = midi_path + "_" + str(i) + ".mid"
            try:
                newscore.write_midi(midi_path_new)
            except ValueError as e:
                error.add(str(e))
                error.add(midi_path_new)
                print("Error: ", midi_path_new)
                flag = 1
            if flag != 1:
                pitch_min,pitch_max=pitch_range(midi_path_new)
                root=root_octave(midi_path_new)
                print("root octave: ",root)
                if pitch_min<min_pitch or pitch_max>max_pitch or \
                        (root!=set([2,3]) and root!=set([2]) and root!=set([3])):
                    os.remove(midi_path_new)
            score = score.transpose(-i)


TIMESIGN={'[6, 8]': 0, '[4, 4]': 1, '[9, 8]': 2, '[2, 4]': 3, '[3, 4]': 4, '[2, 2]': 5, '[6, 4]': 6, '[3, 2]': 7}
DURATION={2:0, 4:1, 6:2, 8:3, 10:4, 12:5, 16:6, 18:7, 20:8, 22:9, 24:10, 30:11, 32:12, 36:13, 42:14, 44:15, 48:16, 54:17, 56:18, 60:19,
          64:20, 66:21, 68:22, 72:23, 78:24, 80:25, 84:26, 90:27, 92:28, 96:29, 102:30, 108:31, 120:32, 126:33, 132:34, 138:35, 144:36}
def melody2repre(melody):
    """
        one-hot melody representation
        0: bar
        1: rest
        2-61: pitch
        62-98: duration
        99-106: time signature
    """
    melody_list=[]
    for element in melody:
        if element == 'b':
            melody_list.append(0)
        elif len(element)==3:
            timeSign=str([element[1], element[2]])
            melody_list.append(TIMESIGN[timeSign]+99)
        else:
            pitch=element[0]
            duration=element[1]
            if pitch==0:
                melody_list.append(1)
            else:
                melody_list.append(pitch-42+2)
            melody_list.append(DURATION[duration]+62)
    return melody_list


INTERVAL={'[3, 4]':1, '[3, 3]':2, '[4, 3]':3, '[3, 4, 3]':4, '[4, 3, 3]':5, '[4, 3, 4]':6}
def chord2repre(chord):
    """
        multi-hot chord representation
        0-6: chord mode
        7-47: root tone
    """
    chord_list=[]
    for element in chord:
        type=[0]*7
        root=[0]*41
        if element==[0]:
            type[0]=1
            root[-1]=1
        else:
            inter=[element[i + 1] - element[i] for i in range(len(element) - 1)]
            type[INTERVAL[str(inter)]]=1
            root[element[0]-30]=1
        type.extend(root)
        chord_list.append(type)
    return chord_list


def twoChordPerBar(path):
    """
        Processing MIDI files into training data, enforcing two chords per bar.
    """
    duration_type=set()
    midi_file = os.listdir(path)
    cnt=0
    wrong_files = set()
    all_melody=[]
    all_chord=[]
    for file in midi_file:
        duration_per_file = set()
        cnt+=1
        print(cnt,file)
        midi_path=os.path.join(path,file)
        chord_list=[];melody_list=[]
        streamm = converter.parse(midi_path)
        melodys=streamm[0]
        chordss=streamm[1]
        melody_part = instrument.partitionByInstrument(melodys)
        chord_part = instrument.partitionByInstrument(chordss)
        melody_notes = melody_part.parts[0].recurse()
        chord_notes = chord_part.parts[0].recurse()
        chord_notes =delTie(chord_notes)
        dur_sum=0
        i=0
        for c_note in chord_notes:
            i+=1
            if isinstance(c_note, meter.TimeSignature):
                bar = (96 * c_note.numerator) // c_note.denominator
                dur_sum=0
            else:
                if len(c_note[0])>1:
                    pitch_value = c_note[0]
                    interval = [pitch_value[i + 1] - pitch_value[i] for i in range(len(pitch_value) - 1)]
                    if str(interval)=='[0, 4, 0, 3, 0]':
                        print(pitch_value)
                        print("wrong interval",midi_path)
                else:
                    pitch_value = [0]
                c_dur=c_note[1]
                if c_dur%(bar//2)==0:
                    chord_list.extend([pitch_value]*int(c_dur//(bar//2)))
                else:
                    if dur_sum%bar==0:
                        if c_dur>bar/2:
                            chord_list.extend([pitch_value] * 2)
                        else:
                            chord_list.append(pitch_value)
                    else:
                        if c_dur>bar/2:
                            chord_list.append(pitch_value)
                        else:
                            if dur_sum%bar<=(bar/2) and (dur_sum+c_dur)%bar>(bar/2):
                                chord_list.append(pitch_value)

                dur_sum+=c_note[1]
        dur_sum = 0
        bar_num=0
        for m_note in melody_notes:
            if isinstance(m_note, meter.TimeSignature):
                melody_list.append(["k",m_note.numerator,m_note.denominator])
                bar = (96 * m_note.numerator) // m_note.denominator
                dur_sum = 0
            else:
                if isinstance(m_note, note.Note) or isinstance(m_note, note.Rest):
                    if dur_sum % bar == 0:
                        melody_list.append("b")
                        bar_num+=1
                    note_duration=int(m_note.duration.quarterLength*24)
                    if isinstance(m_note, note.Note):
                        note_pitch = m_note.pitch.midi
                    else:
                        note_pitch =0
                    if ((dur_sum+note_duration)//bar+1)>bar_num:
                        dur1=bar-(dur_sum%bar)
                        if dur1!=bar:
                            melody_list.append([note_pitch, dur1])
                            duration_per_file.add(dur1)
                            melody_list.append("b")
                            bar_num+=1
                        else:
                            dur1=0
                        for k in range((note_duration-dur1)//bar):
                            melody_list.append([note_pitch, bar])
                            duration_per_file.add(bar)
                            melody_list.append("b")
                            bar_num+=1
                        if (note_duration-dur1)%bar!=0:
                            melody_list.append([note_pitch, note_duration-dur1-((note_duration-dur1)//bar)*bar])
                            duration_per_file.add(note_duration-dur1-((note_duration-dur1)//bar)*bar)
                        else:
                            bar_num-=1
                            melody_list.pop(-1)
                    else:
                        melody_list.append([note_pitch,note_duration])
                        duration_per_file.add(note_duration)
                        if (dur_sum + note_duration) % bar == 0 and (dur_sum + note_duration) // bar > bar_num:
                            bar_num = ((dur_sum + note_duration) // bar - 1)
                        if isinstance(m_note, note.Note) and m_note.pitch.midi!=0 and note_duration==0:
                            print("Wrong:note duration !!!!!")
                    dur_sum+=note_duration
        if melody_list[-1]=='b':
            melody_list.pop()
        bar_num=melody_list.count('b')
        if len(chord_list)!=bar_num*2:
            src = os.path.join(path, file)
            newpath = path[:path.rfind('/') + 1]
            dst = os.path.join(newpath + "del_trans", file)
            shutil.move(src, dst)
            index = file.rfind("_")
            wrong_files.add(file[:index])
        else:
            duration_type = duration_type | duration_per_file
        # assert bar_num==len(chord_list)/2==melody_list.count('b')
        melody_list=melody2repre(melody_list)
        chord_list=chord2repre(chord_list)
        all_melody.append(melody_list)
        all_chord.append(chord_list)
    print("--------------------------------------------------------------------")
    for i in wrong_files:
        print(i)
    temppath=path[:path.rfind('/')+1]
    tempfile=path[path.rfind('/')+1:-6]
    temppath2=temppath[:-1]
    dataset_name=temppath2[temppath2.rfind('/')+1:temppath2.rfind('_')]
    save_melody="../data/"+dataset_name+"_melody_"+tempfile+".data"
    save_chord="../data/"+dataset_name+"_chord_"+tempfile+".data"
    file = open(save_melody, 'wb')
    pickle._dump(all_melody, file)
    file.close()
    file = open(save_chord, 'wb')
    pickle._dump(all_chord, file)
    file.close()
    return


CHORDTYPE={0:'rest',1:'m',2:'dim',3:'maj',4:'m7',5:'7',6:'maj7'}
def computeValence(path):
    """
        compute valence for the chord sequence
    """
    file = open(path, 'rb')
    data = pickle.load(file)
    valence_list=[]
    for i in data:
        valence=[]
        for j in range(len(i)):
            if (j+1)%2==0:
                v1=CHORDTYPE[i[j][0:7].index(1)]
                v2=CHORDTYPE[i[j-1][0:7].index(1)]
                valence.append(calc_chords_val([v1,v2]))
        piece_v=calc_piece_val(valence)
        valence.insert(0,piece_v)
        valence_list.append(valence)
    newpath=path[:path.rfind(".")]+"_valence.data"
    file = open(newpath, 'wb')
    pickle._dump(valence_list, file)
    file.close()
    return


if __name__=='__main__':
    NMD_train="../data/NMD_MIDI/train/"
    NMD_test="../data/NMD_MIDI/test/"
    transpose_midi(NMD_train)
    transpose_midi(NMD_test)
    NMD_MIDI_train="../data/NMD_MIDI/train_trans"
    NMD_MIDI_test="../data/NMD_MIDI/test_trans"
    twoChordPerBar(NMD_MIDI_train)
    twoChordPerBar(NMD_MIDI_test)
    computeValence(path="../data/NMD_MIDI/NMD_chord_train.data")
    computeValence(path="../data/NMD_MIDI/NMD_chord_test.data")