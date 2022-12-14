import glob
import os
from typing import Union
from pathlib import Path
from copy import deepcopy

from tqdm import tqdm

import torch
import librosa
import numpy as np
from scipy.spatial.transform import Rotation as R

from .loader import AISTDataset

import warnings
warnings.filterwarnings("ignore")

def audio_features(audio_dir, audio_name):
    HOP_LENGTH = 512
    SR = 60 * HOP_LENGTH

    def _get_tempo(audio_name):
        """Get tempo (BPM) for a music by parsing music name."""
        assert len(audio_name) == 4
        if audio_name[0:3] in ['mBR', 'mPO', 'mLO', 'mMH', 'mLH', 'mWA', 'mKR', 'mJS', 'mJB']:
            return int(audio_name[3]) * 10 + 80
        elif audio_name[0:3] == 'mHO':
            return int(audio_name[3]) * 5 + 110
        else: assert False, audio_name


    data, _ = librosa.load(os.path.join(audio_dir, f"{audio_name}.wav"), sr=SR)
    envelope = librosa.onset.onset_strength(data, sr=SR)  # (seq_len,)
    mfcc = librosa.feature.mfcc(data, sr=SR, n_mfcc=20).T  # (seq_len, 20)
    chroma = librosa.feature.chroma_cens(
            data, sr=SR, hop_length=HOP_LENGTH, n_chroma=12).T  # (seq_len, 12)

    peak_idxs = librosa.onset.onset_detect(
            onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH)
    peak_onehot = np.zeros_like(envelope, dtype=np.float32)
    peak_onehot[peak_idxs] = 1.0  # (seq_len,)

    tempo, beat_idxs = librosa.beat.beat_track(
            onset_envelope=envelope, sr=SR, hop_length=HOP_LENGTH,
            start_bpm=_get_tempo(audio_name), tightness=100)
    beat_onehot = np.zeros_like(envelope, dtype=np.float32)
    beat_onehot[beat_idxs] = 1.0  # (seq_len,)

    audio_feature = np.concatenate([
            envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]
        ], axis=-1)
    return audio_feature


class Dataloader:
    NATIVE_FPS = 60

    def __init__(
        self, 
        dataset: AISTDataset,
        audio_dir: str,
        config: dict,
        split: str = "train",
        no_preprocessed: bool = False,
        method: str = "2d",
        fps: int = 60,
        return_smpl: bool = False,
        ) -> None:
        
        self.dataset = dataset
        self.audio_dir = audio_dir
        self.config = config
        self.no_preprocessed = no_preprocessed
        self.method = method
        self.fps = fps
        self.return_smpl = return_smpl

        self.seq_len = int(self.config["sequence_length"]*self.NATIVE_FPS/self.fps)
        self.audio_len = int(self.config["audio_length"]*self.NATIVE_FPS/self.fps)
        self.target_len = int(self.config["target_length"]*self.NATIVE_FPS/self.fps)

        self.split = split

        self.indices = list()
        self.dances = dict()
        self.music = dict()
        self.smpl = dict()

        self.individual_length = 0
        
        self._store_in_memory(split)

    
    def _store_in_memory(self, split):
        path = Path(self.dataset.motion_dir).parent
        if split == "train":
            path = path / Path("splits/crossmodal_train.txt")
        elif split == "val":
            path = path / Path("splits/crossmodal_val.txt")
        else:
            path = path / Path("splits/crossmodal_test.txt")
        
        files = np.loadtxt(path, dtype=str)
        ignore_list = np.loadtxt(self.dataset.filter_file, dtype=str)

        for f in tqdm(files):
            # file_path = os.path.join(self.dataset.motion_dir, f + ".pkl")
            if f in ignore_list:
                continue
            self.individual_length += 1
            piece = f
            music = piece.split("_")[-2]
            if self.method == "smpl":
                smpl_poses, smpl_scaling, smpl_trans = self.dataset.load_motion(self.dataset.motion_dir, piece)
                self.smpl[piece] = (deepcopy(smpl_poses), deepcopy(smpl_scaling), deepcopy(smpl_trans))

                smpl_trans /= smpl_scaling
                smpl_poses = R.from_rotvec(
                smpl_poses.reshape(-1, 3)).as_matrix().reshape(smpl_poses.shape[0], -1)

                smpl_motion = np.concatenate([smpl_trans, smpl_poses], axis=-1)
    
                motion = np.pad(smpl_motion, [[0, 0], [6, 0]])

            elif self.method == "2d":
                keypoints2d, _, _ = self.dataset.load_keypoint2d(self.dataset.keypoint2d_dir, piece)
                motion = np.moveaxis(keypoints2d, 0, 1)

                motion[:, :, :, 0] = (motion[:, :, :, 0] - 1920/2)/(1920/2)
                motion[:, :, :, 1] = (motion[:, :, :, 1] - 1080/2)/(1080/2)
                shape = motion.shape
                motion = motion[:, :, :, :2].reshape(shape[0], shape[1], shape[2]*2)

            self.dances[piece] = deepcopy(motion)
            if music not in self.music.keys():
                audio_feature = audio_features(self.audio_dir, music)
                self.music[music] = audio_feature
                
        for music_name, audio in self.music.items():
            audio_frames = audio.shape[0] - self.audio_len

            for dance_name, dance in self.dances.items():
                if music_name not in dance_name:
                    continue

                motion_frames = dance.shape[0] - self.seq_len - self.target_len
                frames = min(audio_frames, motion_frames)
                for i in range(0, frames):

                    if self.method == "2d":
                        for c in range(dance.shape[1]):
                            if np.sum(np.isnan(dance[i: i + self.seq_len + self.target_len, c])) > 0:
                                continue
                            self.indices.append((i, c, music_name, dance_name))
                            if split == "test" or split == "val" and self.no_preprocessed:
                                break
                    else:
                        self.indices.append((i, music_name, dance_name))
                        if split == "test" or split == "val" and self.no_preprocessed:
                            break
    
    def __len__(self):
        if self.no_preprocessed:
            return int(self.individual_length)
        return len(self.indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        if self.method == "2d":
            frame, camera, music_name, dance_name = self.indices[index]    
        else:
            frame, music_name, dance_name = self.indices[index]
        audio = self.music[music_name]
        dance = self.dances[dance_name]

        if self.no_preprocessed:
            if self.return_smpl:
                return self.smpl[dance_name], audio
            return torch.from_numpy(dance).type(torch.float32), torch.from_numpy(audio).type(torch.float32)

        step = int(self.NATIVE_FPS/self.fps)
        if self.method == "2d":
            x = dance[frame:frame + self.seq_len:step, camera]
            y = dance[frame + self.seq_len:frame + self.seq_len + self.target_len:step, camera]
            m = audio[frame:frame + self.audio_len:step]
        else:
            x = dance[frame:frame + self.seq_len:step]
            y = dance[frame + self.seq_len:frame + self.seq_len + self.target_len:step]
            m = audio[frame:frame + self.audio_len:step]

        x, m, y = torch.from_numpy(x).type(torch.float32), torch.from_numpy(m).type(torch.float32), torch.from_numpy(y).type(torch.float32)

        return x, m, y







if __name__ == '__main__':
    dataset = AISTDataset("/home/jon/Documents/dance/data")
    loader = Dataloader(
        dataset, 
        "/home/jon/Documents/dance/data/wav", 
        None, 
        config={"audio_length": 240, "sequence_length": 120, "target_length": 20}, 
        )
    for x, m, y in loader:
        print(x.shape, m.shape, y.shape)