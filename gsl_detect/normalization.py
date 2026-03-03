from loguru import logger
import numpy as np
from pathlib import Path
from tqdm import tqdm 
from gsl_detect.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, FACE_SELECTED_INDICES
import pandas as pd

N_FACE = len(FACE_SELECTED_INDICES)
POSE_END = 132
FACE_END = POSE_END + N_FACE * 3
LH_END = FACE_END + 63
RH_END = LH_END + 63

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

@logger.catch
def pose_relative_normalize(frame):
    pose = frame[:POSE_END].reshape(-1, 3)
    left_shoulder = pose[LEFT_SHOULDER, :3]
    right_shoulder = pose[RIGHT_SHOULDER, :3]
    anchor = (left_shoulder + right_shoulder) / 2.0
    scale = np.linalg.norm(right_shoulder - left_shoulder) 

    if scale > 1e-6:
        return frame #pose not detected, skip normalization
    
    normalized = frame.copy()

    pose_xyz = pose[:, :3]
    pose_xyz = (pose_xyz - anchor) / scale
    pose[:, :3] = pose_xyz
    normalized[:POSE_END] = pose.flatten()

    face = frame[POSE_END:FACE_END].reshape(-1, 3)
    face = (face - anchor) / scale
    normalized[POSE_END:FACE_END] = face.flatten()

    lh = frame[FACE_END:LH_END].reshape(21, 3)
    lh = (lh - anchor) / scale
    normalized[FACE_END:LH_END] = lh.flatten()

    rh = frame[LH_END:RH_END].reshape(21, 3)
    rh = (rh - anchor) / scale
    normalized[LH_END:RH_END] = rh.flatten()

    return normalized

@logger.catch
def pr_normalize_sequence(sequence):
    return np.array([pose_relative_normalize(frame) for frame in sequence])

@logger.catch
def zscore_stats(train_files):
    n = 0
    mean = None
    M2 = None  # Welford's algorithm for variance

    for f in tqdm(train_files, desc="Computing mean/std"):
        seq = np.load(f)
        seq = pr_normalize_sequence(seq)
        if mean is None:
            mean = np.zeros(seq.shape[1])
            M2 = np.zeros(seq.shape[1])

        for frame in seq:
            n += 1
            delta = frame - mean
            mean += delta / n
            M2 += delta * (frame - mean)

    std = np.sqrt(M2 / n)
    std[std < 1e-6] = 1.0
    return mean, std

@logger.catch
def apply_normalization(files, df, mean, std, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True) 

    for f in tqdm(files, desc=f"Normalizing {output_dir.name}"):

        stem = f.stem.replace("_proc", "")          # s17_g201_r2_rgb
        signer_folder = f.parent.name               # signer17
        video_key = f"{signer_folder}/{stem}"       # signer17/s17_g201_r2_rgb

        match = df[df["Video"] == video_key]
        if match.empty:
            logger.warning(f"No matching video found for {f}, skipping.")
            continue
        video_rel = match.iloc[0]["Video"]
        out_file = output_dir / f"{video_rel.replace('.npy', '_norm.npy')}"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        seq = np.load(f)
        seq = pr_normalize_sequence(seq)
        seq = (seq - mean) / std
        np.save(out_file, seq)

@logger.catch
def get_files(df, signers, interim_dir=INTERIM_DATA_DIR):
    rows = df[df["Signer"].isin(signers)]
    return [interim_dir / f"{row['Video']}_proc.npy" for _, row in rows.iterrows() if (interim_dir / f"{row['Video']}_proc.npy").exists()]

if __name__ == "__main__":
    df = pd.read_csv(RAW_DATA_DIR / "isolated_GSL_corpus.csv")
    df["Signer"] = df["Video"].apply(lambda x: x.split("/")[0])
    all_signers = df["Signer"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(all_signers)
    test_signers = all_signers[:3] # first 3 signers for testing
    val_signers = all_signers[3:6] # next 3 signers for validation
    train_signers = all_signers[6:] # all remaining signers for training

    train_files = get_files(df, train_signers)
    val_files = get_files(df, val_signers)
    test_files = get_files(df, test_signers)

    mean, std = zscore_stats(train_files)
    np.save(PROCESSED_DATA_DIR / "mean.npy", mean)
    np.save(PROCESSED_DATA_DIR / "std.npy", std)
    apply_normalization(train_files, df, mean, std, PROCESSED_DATA_DIR / "train")
    apply_normalization(val_files, df, mean, std, PROCESSED_DATA_DIR / "val")
    apply_normalization(test_files, df, mean, std, PROCESSED_DATA_DIR / "test")

    print("Normalization complete. Mean and std saved to processed directory.")