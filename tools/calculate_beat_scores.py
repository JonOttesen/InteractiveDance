import os
import time
from librosa import beat
import torch
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
import scipy.signal as scisignal

from dance.loaders.loader import AISTDataset
from dance.loaders.dataloader import Dataloader

from dance.models.fact.fact import FACTModel
from dance.models.fact.config import audio_config, fact_model, motion_config, multi_model_config


def eye(n, batch_shape):
    iden = np.zeros(np.concatenate([batch_shape, [n, n]]))
    iden[..., 0, 0] = 1.0
    iden[..., 1, 1] = 1.0
    iden[..., 2, 2] = 1.0
    return iden


def get_closest_rotmat(rotmats):
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
    iden = eye(3, det.shape)
    iden[..., 2, 2] = np.sign(det)
    r_closest = np.matmul(np.matmul(u, iden), vh)
    return r_closest


def recover_to_axis_angles(motion):
    batch_size, seq_len, dim = motion.shape
    assert dim == 225
    transl = motion[:, :, 6:9]
    rotmats = get_closest_rotmat(
        np.reshape(motion[:, :, 9:], (batch_size, seq_len, 24, 3, 3))
    )
    axis_angles = R.from_matrix(
        rotmats.reshape(-1, 3, 3)
    ).as_rotvec().reshape(batch_size, seq_len, 24, 3)
    return axis_angles, transl


def recover_motion_to_keypoints(motion, smpl_model):
    smpl_poses, smpl_trans = recover_to_axis_angles(motion)
    smpl_poses = np.squeeze(smpl_poses, axis=0)  # (seq_len, 24, 3)
    smpl_trans = np.squeeze(smpl_trans, axis=0)  # (seq_len, 3)
    keypoints3d = smpl_model.forward(
        global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
        body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
        transl=torch.from_numpy(smpl_trans).float(),
    ).joints.detach().numpy()[:, :24, :]   # (seq_len, 24, 3)

    output = smpl_model.forward(
        global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
        body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
        transl=torch.from_numpy(smpl_trans).float(),
    )

    plot(output, smpl_model)

    return keypoints3d


def motion_peak_onehot(joints):
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

def plot(output, model):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices_video = output.vertices.detach().cpu().numpy().squeeze()
    joints_video = output.joints.detach().cpu().numpy().squeeze()[:, :24, :]


    frames = vertices_video.shape[0]
    for i in range(frames):
        vertices = vertices_video[i]
        joints = joints_video[i]
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_axis_off()
        ax.view_init(elev=-90, azim=90)

        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 2)
        ax.set_zlim(-1, 1)

        mesh = Poly3DCollection(vertices[model.faces], alpha=0.1)
        face_color = (1.0, 1.0, 0.9)
        edge_color = (0, 0, 0)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')

        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], alpha=0.1)
        plt.savefig("dances/{:03d}.png".format(i), dpi=150)
        plt.close()

def alignment_score(music_beats, motion_beats, sigma=3):
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


def main():
    import glob
    from tqdm import tqdm
    from smplx import SMPL

    audio_config.transformer.intermediate_size = 1024
    motion_config.transformer.intermediate_size = 1024
    multi_model_config.transformer.intermediate_size = 1024
    multi_model_config.transformer.num_hidden_layers =  4

    model = FACTModel(audio_config, motion_config, multi_model_config, pred_length=20)
    model = model.to("cuda:0")
    model.eval()

    model_path = model_path = "/home/jon/Documents/dance/checkpoint-best.pth"
    model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'])

    # set smpl
    smpl = SMPL(model_path="/home/jon/Documents/dance/smpl/models/", gender='MALE', batch_size=1)

    path = "/home/jon/Documents/dance/data/"

    # calculate score on real data
    dataset = AISTDataset(path)

    loader = Dataloader(
        dataset, 
        "/home/jon/Documents/dance/data/wav", 
        None, 
        config={"audio_length": 240, "sequence_length": 120, "target_length": 20}, 
        keypoint_dir="motions",
        split="test",
        no_preprocessed=True,
        return_smpl=True,
        )
    print(len(loader))
    exit()

    beat_scores = []
    for i, (motion, audio) in enumerate(loader):
        # get real data motion beats
        smpl_poses, smpl_scaling, smpl_trans = motion
        smpl_trans /= smpl_scaling


        keypoints3d = smpl.forward(
            global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
            body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
            transl=torch.from_numpy(smpl_trans).float(),
        ).joints.detach().numpy()[:, :24, :]   # (seq_len, 24, 3)

        output = smpl.forward(
            global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
            body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
            transl=torch.from_numpy(smpl_trans).float(),
        )

        motion_beats = motion_peak_onehot(keypoints3d)
        # get real data music beats

        audio_beats = audio[:keypoints3d.shape[0], -1] # last dim is the music beats

        # get beat alignment scores
        beat_score = alignment_score(audio_beats, motion_beats, sigma=3)
        beat_scores.append(beat_score)
    n_samples = i + 1
    print ("\nBeat score on real data: %.3f\n" % (sum(beat_scores) / n_samples))

    beat_scores = []
    # for i, (motion, audio) in enumerate(loader):
        # get real data motion beats
        # smpl_poses, smpl_scaling, smpl_trans = motion
        # smpl_trans /= smpl_scaling
    loader.no_preprocessed = False
    for i, (motion, audio, _) in enumerate(loader):
        # get real data motion beats
        motion = motion[:120]

        inp = {"motion_input": motion.unsqueeze(0).to("cuda:0"), "audio_input": audio.unsqueeze(0).to("cuda:0")}
        with torch.no_grad():
            start = time.time()
            pred = model.infer_auto_regressive(inp, steps=1200).cpu().numpy()[0]
            print("Total time: ", time.time() - start)

        result_motion = np.expand_dims(np.concatenate([
            motion,
            pred
            ], axis=0), axis=0)  # [1, 120 + 1200, 225]

        keypoints3d = recover_motion_to_keypoints(result_motion, smpl)

        motion_beats = motion_peak_onehot(keypoints3d)
        
        audio_beats = audio[:, -1] # last dim is the music beats
        beat_score = alignment_score(audio_beats[120:], motion_beats[120:], sigma=3)
        beat_scores.append(beat_score)
    print ("\nBeat score on generated data: %.3f\n" % (sum(beat_scores) / n_samples))

if __name__=="__main__":
    main()