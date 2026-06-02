# Online augmentation of the landmark data -- see different versions of the same array in each epoch with randomness
import numpy as np

# Feature layout (no face landmarks)
POSE_END = 60    # 15 pts × 4 (x, y, z, visibility)
LH_END   = 123   # + 21 pts × 3
RH_END   = 186   # + 21 pts × 3


def gaussian_noise(sequence, std=0.02):
    noise = np.random.randn(*sequence.shape) * std
    return sequence + noise


def temporal_warp(sequence, rate_range=(0.8, 1.2)):
    n_frames, n_feats = sequence.shape
    rate   = np.random.uniform(*rate_range)
    new_len = int(n_frames * rate)
    new_len = max(2, new_len)

    old_idx = np.linspace(0, n_frames - 1, new_len)
    new_seq = np.zeros((new_len, n_feats))
    for f in range(n_feats):
        new_seq[:, f] = np.interp(old_idx, np.arange(n_frames), sequence[:, f])

    if new_len >= n_frames:
        return new_seq[:n_frames]
    else:
        pad = np.zeros((n_frames - new_len, n_feats))
        return np.vstack([new_seq, pad])


def horizontal_flip(sequence):
    seq = sequence.copy()

    # Pose: flip x (col 0 of every 4-tuple)
    pose = seq[:, :POSE_END].reshape(-1, 15, 4)
    pose[:, :, 0] = 1.0 - pose[:, :, 0]
    seq[:, :POSE_END] = pose.reshape(-1, POSE_END)

    # Hands: flip x και swap lh↔rh
    lh = seq[:, POSE_END:LH_END].copy().reshape(-1, 21, 3)
    rh = seq[:, LH_END:RH_END].copy().reshape(-1, 21, 3)
    lh[:, :, 0] = 1.0 - lh[:, :, 0]
    rh[:, :, 0] = 1.0 - rh[:, :, 0]
    seq[:, POSE_END:LH_END] = rh.reshape(-1, 63)  # rh → lh slot
    seq[:, LH_END:RH_END]   = lh.reshape(-1, 63)  # lh → rh slot

    return seq


def random_rotation_z(sequence, max_angle_deg=15):
    angle = np.radians(np.random.uniform(-max_angle_deg, max_angle_deg))
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    def rotate_xyz(pts):
        # pts: (n_frames, n_pts, 3+)
        x = pts[:, :, 0] * cos_a - pts[:, :, 1] * sin_a
        y = pts[:, :, 0] * sin_a + pts[:, :, 1] * cos_a
        pts = pts.copy()
        pts[:, :, 0] = x
        pts[:, :, 1] = y
        return pts

    seq = sequence.copy()

    pose = seq[:, :POSE_END].reshape(-1, 15, 4)
    pose = rotate_xyz(pose)
    seq[:, :POSE_END] = pose.reshape(-1, POSE_END)

    lh = seq[:, POSE_END:LH_END].reshape(-1, 21, 3)
    lh = rotate_xyz(lh)
    seq[:, POSE_END:LH_END] = lh.reshape(-1, 63)

    rh = seq[:, LH_END:RH_END].reshape(-1, 21, 3)
    rh = rotate_xyz(rh)
    seq[:, LH_END:RH_END] = rh.reshape(-1, 63)

    return seq


def random_scale(sequence, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(*scale_range)
    return sequence * scale


def augment(sequence, p=0.5):
    if np.random.rand() < p:
        sequence = gaussian_noise(sequence, std=0.02)
    if np.random.rand() < p:
        sequence = temporal_warp(sequence, rate_range=(0.8, 1.2))
    if np.random.rand() < p:
        sequence = horizontal_flip(sequence)
    if np.random.rand() < p:
        sequence = random_rotation_z(sequence, max_angle_deg=10)
    if np.random.rand() < p:
        sequence = random_scale(sequence, scale_range=(0.9, 1.1))
    return sequence