import pickle
import numpy as np
from remi2midi import remi2midi
from dataloader import REMIFullSongTransformerDataset
from utils import pickle_load
import random
import sys
import os
import yaml
sys.path.append('./model')

with open('remi_dataset/836.pkl', 'rb') as f:
    piece = pickle.load(f)

def word2event(word_seq, idx2event):
  return [ idx2event[w] for w in word_seq ]

def check():
    config_path = './config/default.yaml'
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    data_split = 'pickles/test_pieces.pkl'

    data_dir = config['data']['data_dir']
    vocab_path = config['data']['vocab_path']

    dset = REMIFullSongTransformerDataset(
        data_dir, vocab_path, 
        do_augment=False,
        model_enc_seqlen=config['data']['enc_seqlen'], 
        model_dec_seqlen=config['generate']['dec_seqlen'],
        model_max_bars=config['generate']['max_bars'],
        pieces=pickle_load(data_split),
        pad_to_same=False
    )

    pieces = random.sample(range(len(dset)), 1)
    for p in pieces:
    p_data = dset[p]
    p_id = p_data['piece_id']
    p_bar_id = p_data['st_bar_id']
    p_data['enc_input'] = p_data['enc_input'][ : p_data['enc_n_bars'] ]
    p_data['enc_padding_mask'] = p_data['enc_padding_mask'][ : p_data['enc_n_bars'] ]
    orig_song = p_data['dec_input'].tolist()[:p_data['length']]
    orig_song = word2event(orig_song, dset.idx2event)
    
    midi_obj, orig_tempo = remi2midi(orig_song, 'midi_files.mid', return_first_tempo=True, enforce_tempo=False)

