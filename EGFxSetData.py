from torch.utils.data import Dataset
from torch import from_numpy
import torchaudio 
from pedalboard_utils import * 

import os 
from pathlib import Path

class EGFxSetData(Dataset): 
    def __init__(self, effects_probs=None):
        self.data_path = Path("/home/yuc3/guitar_effects/EGFxSetResampled")
        self.wav_paths = list(self.data_path.glob("*/*.wav"))
        self.meta_data = [] 
        self.effects_probs = effects_probs
        
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

        if self.effects_probs is not None:
            pedalboard = get_random_board(self.effects_probs)
            waveform = from_numpy(pedalboard(waveform.numpy(), sr, reset=False))
            board_string = get_board_string(pedalboard) 
        
        return (waveform, sr), board_string
    def __len__(self):
        return self.n_samples