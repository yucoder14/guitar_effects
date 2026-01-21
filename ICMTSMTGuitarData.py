from torch.utils.data import Dataset
from torch import from_numpy
import torchaudio 
from pedalboard_utils import * 

import os 
from pathlib import Path

# ICMT-SMT-Guitar data offers guitar samples that are monophonic and polyphonic
# it also offers smaples with effects applied, but I'll be working mostly with 
# NoFx samples

IDMT = Path("/home/yuc3/guitar_effects/IDMT-SMT-AUDIO-EFFECTS/IDMT-SMT-AUDIO-EFFECTS")

class ICMTSMTGuitarDataMono(Dataset): 
    def __init__(self, effects_probs=None, target_length_seconds=2.0):
        self.data_path = IDMT / "Gitarre monophon" 
        self.wav_paths = [path for path in list(self.data_path.glob("*/*/*.wav")) if path.parent.name == "NoFX"]
        self.meta_data = [] 
        self.effects_probs = effects_probs
        self.sr = 44100
        self.target_samples = int(target_length_seconds * self.sr)
        
        for wav_path in self.wav_paths:
            codes = wav_path.name.split(".")[0].split("-")
            id_num = codes[-1] 
            
            instrument_type = codes[0][0] 
            instrument_brand = codes[0][1]
            playing_style = codes[0][2] 
            
            midi_number = codes[1][:2]
            string_number = codes[1][2]
            fret_number = codes[1][3:]
            
            self.meta_data.append({
                "path": str(wav_path), 
                "id": id_num,
                "midi": midi_number,
                "string": string_number, 
                "fret": fret_number,
                "instrument": instrument_type,
                "brand": instrument_brand,
                "playing_style": playing_style
            })
        self.n_samples = len(self.wav_paths)
    def __getitem__(self, index):
        meta_data = self.meta_data[index]
        waveform, sr = torchaudio.load(meta_data["path"])
        board_string = "clean"

        # apply effects 
        if self.effects_probs is not None:
            pedalboard = get_random_board(self.effects_probs)
            waveform = from_numpy(pedalboard(waveform.numpy(), sr, reset=False))
            board_string = get_board_string(pedalboard) 

        # crop
        current_len = waveform.shape[-1]
        if current_len < self.target_samples:
            padding = self.target_samples - current_len
            aug_waveform = pad(aug_waveform, (0, padding))
            current_len = aug_waveform.shape[-1]

        max_start = current_len - self.target_samples
        
        if max_start <= 0:
            i = 0
        else:
            i = random.randint(0, max_start)
            
        return (waveform[..., i : i + self.target_samples], sr), board_string
        
    def __len__(self):
        return self.n_samples

class ICMTSMTGuitarDataPoly(Dataset): 
    def __init__(self, effects_probs=None, target_length_seconds=2.0):
        self.data_path = IDMT / "Gitarre polyphon" 
        self.wav_paths = [path for path in list(self.data_path.glob("*/*/*.wav")) if path.parent.name == "NoFX"]
        self.meta_data = [] 
        self.effects_probs = effects_probs
        self.sr = 44100
        self.target_samples = int(target_length_seconds * self.sr)
        
        for wav_path in self.wav_paths:
            codes = wav_path.name.split(".")[0].split("-")
            id_num = codes[-1] 
            
            instrument_type = codes[0][0] 
            instrument_brand = codes[0][1]
            playing_style = codes[0][2] 
        
            midi_number = codes[1][:2] 
            chord_type = codes[1][2:4] 
           
            self.meta_data.append({
                "path": str(wav_path), 
                "id": id_num,
                "midi": midi_number,
                "chord_type": chord_type, 
                "instrument": instrument_type,
                "brand": instrument_brand,
                "playing_style": playing_style
            })
        self.n_samples = len(self.wav_paths)
    def __getitem__(self, index):
        meta_data = self.meta_data[index]
        waveform, sr = torchaudio.load(meta_data["path"])
        board_string = "clean"

        # apply effects 
        if self.effects_probs is not None:
            pedalboard = get_random_board(self.effects_probs)
            waveform = from_numpy(pedalboard(waveform.numpy(), sr, reset=False))
            board_string = get_board_string(pedalboard) 

        # crop
        current_len = waveform.shape[-1]
        if current_len < self.target_samples:
            padding = self.target_samples - current_len
            aug_waveform = pad(aug_waveform, (0, padding))
            current_len = aug_waveform.shape[-1]

        max_start = current_len - self.target_samples
        
        if max_start <= 0:
            i = 0
        else:
            i = random.randint(0, max_start)
            
        return (waveform[..., i : i + self.target_samples], sr), board_string
        
    def __len__(self):
        return self.n_samples
