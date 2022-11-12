import os
import time
from librosa import beat
import torch
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
import scipy.signal as scisignal
from scipy import linalg

try:
    from aist_plusplus.features.kinetic import extract_kinetic_features
    from aist_plusplus.features.manual import extract_manual_features
except:
    pass


class Scores(object):

    def __init__(self, smpl) -> None:
        self.smpl = smpl

        self.pred_kinetic = list()
        self.real_kinetic = list()

        self.pred_manual = list()
        self.real_manual = list()
    
    def beat_align(self, keypoints3d, audio):
        # Beat align
        # keypoints3d = self.recover_motion_to_keypoints(motion)
        motion_beats = self.motion_peak_onehot(keypoints3d)
        
        audio_beats = audio[:, -1]
        beat_score = self.alignment_score(audio_beats[120:], motion_beats[120:], sigma=3)
        return beat_score
    
    def accumulate_fid(self, pred_keypoints, real_keypoints):
        self.pred_kinetic.append(extract_kinetic_features(pred_keypoints))
        self.real_kinetic.append(extract_kinetic_features(real_keypoints))

        self.pred_manual.append(extract_manual_features(pred_keypoints))
        self.real_manual.append(extract_manual_features(real_keypoints))
    
    def fid(self):
        FID_k, Dist_k = self.calculate_frechet_feature_distance(self.real_kinetic, self.pred_kinetic)
        FID_g, Dist_g = self.calculate_frechet_feature_distance(self.real_manual, self.pred_manual)

        self.pred_kinetic = list()
        self.real_kinetic = list()

        self.pred_manual = list()
        self.real_manual = list()

        return FID_k, FID_g, Dist_k, Dist_g
    
    def kinetic_fid(self, pred_keypoints, real_keypoints):
        # keypoints3d = self.recover_motion_to_keypoints(pred_motion)
        pred_kinetic = extract_kinetic_features(pred_keypoints)

        # keypoints3d = self.recover_motion_to_keypoints(real_motion)
        real_kinetic = extract_kinetic_features(real_keypoints)

        FID_k, Dist_k = self.calculate_frechet_feature_distance(real_kinetic, pred_kinetic)
        return FID_k, Dist_k

    def manual_fid(self, pred_keypoints, real_keypoints):
        # keypoints3d = self.recover_motion_to_keypoints(pred_motion)
        pred_manual = extract_manual_features(pred_keypoints)

        # keypoints3d = self.recover_motion_to_keypoints(real_motion)
        real_manual = extract_manual_features(real_keypoints)
        FID_g, Dist_g = self.calculate_frechet_feature_distance(real_manual, pred_manual)
        return FID_g, Dist_g

    def eye(self, n, batch_shape):
        iden = np.zeros(np.concatenate([batch_shape, [n, n]]))
        iden[..., 0, 0] = 1.0
        iden[..., 1, 1] = 1.0
        iden[..., 2, 2] = 1.0
        return iden

    def get_closest_rotmat(self, rotmats):
        """
        Finds the rotation matrix that is closest to the inputs in terms of the Frobenius norm. For each input matrix
        it computes the SVD as R = USV' and sets R_closest = UV'. Additionally, it is made sure that det(R_closest) == 1.
        Args:
            rotmats: np array of shape (..., 3, 3).
        Returns:
            A numpy array of the same shape as the inputs.
        """
        u, s, vh = np.linalg.svd(rotmats)
        r_closest = np.matmul(u, vh)

        # if the determinant of UV' is -1, we must flip the sign of the last column of u
        det = np.linalg.det(r_closest)  # (..., )
        iden = self.eye(3, det.shape)
        iden[..., 2, 2] = np.sign(det)
        r_closest = np.matmul(np.matmul(u, iden), vh)
        return r_closest


    def recover_to_axis_angles(self, motion):
        batch_size, seq_len, dim = motion.shape
        assert dim == 225
        transl = motion[:, :, 6:9]
        rotmats = self.get_closest_rotmat(
            np.reshape(motion[:, :, 9:], (batch_size, seq_len, 24, 3, 3))
        )
        axis_angles = R.from_matrix(
            rotmats.reshape(-1, 3, 3)
        ).as_rotvec().reshape(batch_size, seq_len, 24, 3)
        return axis_angles, transl


    def recover_motion_to_keypoints(self, motion, device):
        smpl_poses, smpl_trans = self.recover_to_axis_angles(motion)
        smpl_poses = np.squeeze(smpl_poses, axis=0)  # (seq_len, 24, 3)
        smpl_trans = np.squeeze(smpl_trans, axis=0)  # (seq_len, 3)
        keypoints3d = self.smpl.forward(
            global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float().to(device),
            body_pose=torch.from_numpy(smpl_poses[:, 1:]).float().to(device),
            transl=torch.from_numpy(smpl_trans).float().to(device),
        ).joints.detach().cpu().numpy()[:, :24, :]   # (seq_len, 24, 3)

        return keypoints3d


    def motion_peak_onehot(self, joints):
        """Calculate motion beats.
        Kwargs:
            joints: [nframes, njoints, 3]
        Returns:
            - peak_onhot: motion beats.
        """
        # Calculate velocity.
        velocity = np.zeros_like(joints, dtype=np.float32)
        velocity[1:] = joints[1:] - joints[:-1]
        velocity_norms = np.linalg.norm(velocity, axis=2)
        envelope = np.sum(velocity_norms, axis=1)  # (seq_len,)

        # Find local minima in velocity -- beats
        peak_idxs = scisignal.argrelextrema(envelope, np.less, axis=0, order=10) # 10 for 60FPS
        peak_onehot = np.zeros_like(envelope, dtype=bool)
        peak_onehot[peak_idxs] = 1

        # # Second-derivative of the velocity shows the energy of the beats
        # peak_energy = np.gradient(np.gradient(envelope)) # (seq_len,)
        # # optimize peaks
        # peak_onehot[peak_energy<0.001] = 0
        return peak_onehot


    def alignment_score(self, music_beats, motion_beats, sigma=3):
        """Calculate alignment score between music and motion."""
        if motion_beats.sum() == 0:
            return 0.0
        music_beat_idxs = np.where(music_beats)[0]
        motion_beat_idxs = np.where(motion_beats)[0]
        score_all = []
        for motion_beat_idx in motion_beat_idxs:
            dists = np.abs(music_beat_idxs - motion_beat_idx).astype(np.float32)
            ind = np.argmin(dists)
            score = np.exp(- dists[ind]**2 / 2 / sigma**2)
            score_all.append(score)
        return sum(score_all) / len(score_all)

    def calculate_frechet_feature_distance(self, feature_list1, feature_list2):
        feature_list1 = np.stack(feature_list1)
        feature_list2 = np.stack(feature_list2)

        # normalize the scale
        mean = np.mean(feature_list1, axis=0)
        std = np.std(feature_list1, axis=0) + 1e-10
        feature_list1 = (feature_list1 - mean) / std
        feature_list2 = (feature_list2 - mean) / std

        frechet_dist = self.calculate_frechet_distance(
            mu1=np.mean(feature_list1, axis=0), 
            sigma1=np.cov(feature_list1, rowvar=False),
            mu2=np.mean(feature_list2, axis=0), 
            sigma2=np.cov(feature_list2, rowvar=False),
        )
        avg_dist = self.calculate_avg_distance(feature_list2)
        return frechet_dist, avg_dist

    def calculate_avg_distance(self, feature_list, mean=None, std=None):
        feature_list = np.stack(feature_list)
        n = feature_list.shape[0]
        # normalize the scale
        if (mean is not None) and (std is not None):
            feature_list = (feature_list - mean) / std
        dist = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist += np.linalg.norm(feature_list[i] - feature_list[j])
        dist /= (n * n - n) / 2
        return dist

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.

        Code apapted from https://github.com/mseitzer/pytorch-fid

        Copyright 2018 Institute of Bioinformatics, JKU Linz
        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at
          http://www.apache.org/licenses/LICENSE-2.0
        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        See the License for the specific language governing permissions and
        limitations under the License.

        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        mu and sigma are calculated through:
        ```
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        ```
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                print('Above threshold Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)