import numpy as np
import cv2
from pathlib import Path
from gsl_detect.config import PROCESSED_DATA_DIR

# Feature layout (no face landmarks)
# Pose: 15 pts × 4 coords = 60
# Left hand: 21 pts × 3 coords = 63
# Right hand: 21 pts × 3 coords = 63
# Total: 186
POSE_END = 15 * 4   # 60
LH_END   = POSE_END + 63   # 123
RH_END   = LH_END + 63     # 186


def dataset_stats():
    files = sorted(PROCESSED_DATA_DIR.rglob("*.npz"))
    print(f"\nTotal .npz files found: {len(files)}")

    zero_ratios, frame_counts = [], []

    for f in files:
        data = np.load(f)["sequence"]  # shape: (n_frames, 186)
        assert data.shape[1] == RH_END, \
            f"Expected {RH_END} features, got {data.shape[1]} in {f.name}"
        frame_counts.append(len(data))
        zero_ratio = np.mean(data == 0)
        zero_ratios.append(zero_ratio)

    short = [(f, np.load(f)["sequence"].shape[0]) for f in files if np.load(f)["sequence"].shape[0] > 80]
    print(f"\nFiles with >80 frames: {len(short)}")

    print(f"Feature dims  — expected: {RH_END} (pose=60, lh=63, rh=63)")
    print(f"Frame counts  — min: {min(frame_counts)}, max: {max(frame_counts)}, mean: {np.mean(frame_counts):.1f}")
    print(f"Zero ratio    — min: {min(zero_ratios):.2%}, max: {max(zero_ratios):.2%}, mean: {np.mean(zero_ratios):.2%}")

    bad = [f for f, z in zip(files, zero_ratios) if z > 0.5]
    print(f"\nSuspicious files (>50% zeros): {len(bad)}")


def sample_stats(sample_path):
    data = np.load(sample_path)["sequence"]
    print(f"\nFile: {sample_path.name}")
    print(f"Shape: {data.shape}  (frames × features)  — expected (?, 186)")

    pose = data[:, :POSE_END].reshape(-1, 15, 4)
    lh   = data[:, POSE_END:LH_END].reshape(-1, 21, 3)
    rh   = data[:, LH_END:RH_END].reshape(-1, 21, 3)

    def zero_pct(arr):
        return np.mean(np.all(arr == 0, axis=-1)) * 100

    print(f"Pose  zeros: {zero_pct(pose):.1f}% of frames")
    print(f"Left  hand zeros: {zero_pct(lh):.1f}% of frames")
    print(f"Right hand zeros: {zero_pct(rh):.1f}% of frames")
    print(f"Coordinate range: [{data.min():.3f}, {data.max():.3f}]")


def animate_sample(sample_path, fps=30):
    data = np.load(sample_path)["sequence"]
    delay = int(1000 / fps)

    for i, frame in enumerate(data):
        pose = frame[:POSE_END].reshape(15, 4)
        lh   = frame[POSE_END:LH_END].reshape(21, 3)
        rh   = frame[LH_END:RH_END].reshape(21, 3)

        canvas = np.zeros((480, 848, 3), dtype=np.uint8)

        def draw(landmarks, color):
            for lm in landmarks:
                if not np.all(lm[:2] == 0):
                    x, y = int(lm[0] * 848), int(lm[1] * 480)
                    if 0 <= x < 848 and 0 <= y < 480:
                        cv2.circle(canvas, (x, y), 4, color, -1)

        draw(pose[:, :3], (0, 255, 0))    # green — pose
        draw(lh,          (255, 80, 80))  # blue  — left hand
        draw(rh,          (80, 80, 255))  # red   — right hand

        cv2.putText(canvas, f"Frame {i}/{len(data)-1}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Animation", canvas)
        if cv2.waitKey(delay) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    dataset_stats()

    sample = PROCESSED_DATA_DIR / "train/signer01/s01_g002_r0_rgb_norm.npz"
    sample_stats(sample)
    # animate_sample(sample)
