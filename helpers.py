import os
import random
from note_seq import drums_encoder_decoder
import note_seq
from note_seq import drums_lib
from note_seq import midi_io
from note_seq.protobuf import music_pb2
from note_seq import sequences_lib
import numpy as np

class notLong(Exception):
    pass

def pick_drummer(numbers = range(1,10)):
    return "drummer{}".format(random.choice(numbers))


def random_midi():
    current = os.getcwd()
    path = os.path.join(current, "data") 
    path = os.path.join(path, pick_drummer()) 
    
    ## choose session
    path = os.path.join(path, random.choice(os.listdir(path))) 
    ## choose midi
    path = os.path.join(path, random.choice(os.listdir(path))) 
    return path

def midi_to_input(midi_dir):
    ##capped to 64
    sequence = note_seq.midi_file_to_note_sequence(midi_dir)
    drums = drums_lib.DrumTrack()
    quantized_sequence = sequences_lib.quantize_note_sequence(
            sequence, 4)
    drums.from_quantized_sequence(quantized_sequence, search_start_step=0)
    enc = drums_encoder_decoder.MultiDrumOneHotEncoding()
    data = np.zeros(576)
    tracker = 0
    if len(list(drums)) <=64:
        raise notLong("not loung enough")
    for note in drums[0:64]:
        event_int = enc.encode_event(note)
        event_list = [int(x) for x in '{:09b}'.format(event_int)]
#         print(event_list)
        for byte in event_list:
            data[tracker] = byte
            tracker+=1
    return data

def midi_to_input_uncap(midi_dir):
    sequence = note_seq.midi_file_to_note_sequence(midi_dir)
    drums = drums_lib.DrumTrack()
    quantized_sequence = sequences_lib.quantize_note_sequence(
            sequence, 4)
    drums.from_quantized_sequence(quantized_sequence, search_start_step=0)
    enc = drums_encoder_decoder.MultiDrumOneHotEncoding()
    data = np.array([])
    if len(list(drums)) <=64:
        raise notLong("not loung enough")
    for note in drums:
        holder = np.zeros(9)
        event_int = enc.encode_event(note)
        event_list = [int(x) for x in '{:09b}'.format(event_int)]
        for idx,byte in enumerate(event_list):
            holder[idx] = byte
        data = np.concatenate((data,holder))
    data = data.reshape((-1,9))
    ## end codon
    data = np.vstack((data,np.ones((1,9))))
#     print(data[0])
        
            
    return data
        
        
    