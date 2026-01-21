from torch.utils.data import Dataset
from torch import from_numpy
import torchaudio 
from pedalboard_utils import * 
from torch.nn.functional import pad

import os 
from pathlib import Path

class EGFxSetData(Dataset): 
    def __init__(self, effects_probs=None, target_length_seconds=2.0):
        self.data_path = Path("/home/yuc3/guitar_effects/EGFxSetResampled")
        self.wav_paths = list(self.data_path.glob("*/*.wav"))
        self.meta_data = [] 
        self.effects_probs = effects_probs
        self.sr = 44100
        self.target_samples = int(target_length_seconds * self.sr)
        
        for wav_path in self.wav_paths: 
            pickup = wav_path.parent.name 

            string, fret = wav_path.name.split(".")[0].split("-")
            self.meta_data.append({
                "path": str(wav_path), 
                "pickup": pickup,
                "string": string, 
                "fret": fret
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