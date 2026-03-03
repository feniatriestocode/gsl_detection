import numpy as np
import cv2
from pathlib import Path
from gsl_detect.config import PROCESSED_DATA_DIR, FACE_SELECTED_INDICES

N_FACE = len(FACE_SELECTED_INDICES)
POSE_END = 132
FACE_END = POSE_END + N_FACE * 3
LH_END   = FACE_END + 63
RH_END   = LH_END + 63


def dataset_stats():
    files = sorted(PROCESSED_DATA_DIR.rglob("*.npy"))
    print(f"\nTotal .npy files found: {len(files)}")

    zero_ratios, frame_counts = [], []

    for f in files:
        data = np.load(f)
        frame_counts.append(len(data))
        zero_ratio = np.mean(data == 0)
        zero_ratios.append(zero_ratio)

    print(f"Frame counts  — min: {min(frame_counts)}, max: {max(frame_counts)}, mean: {np.mean(frame_counts):.1f}")
    print(f"Zero ratio    — min: {min(zero_ratios):.2%}, max: {max(zero_ratios):.2%}, mean: {np.mean(zero_ratios):.2%}")

    # Flag suspicious files (>50% zeros = detection likely failing) -- this does not mean anything rn because the signers are shown from the waist up
    bad = [f for f, z in zip(files, zero_ratios) if z > 0.5]
    print(f"\nSuspicious files (>50% zeros): {len(bad)}")
    for f in bad[:100]:  # show first 10
        print(f"  {f}")


def sample_stats(sample_path):
    data = np.load(sample_path)
    print(f"\nFile: {sample_path.name}")
    print(f"Shape: {data.shape}  (frames x features)")

    pose = data[:, :POSE_END].reshape(-1, 33, 4)
    face = data[:, POSE_END:FACE_END].reshape(-1, N_FACE, 3)
    lh   = data[:, FACE_END:LH_END].reshape(-1, 21, 3)
    rh   = data[:, LH_END:RH_END].reshape(-1, 21, 3)

    def zero_pct(arr):
        return np.mean(np.all(arr == 0, axis=-1)) * 100

    print(f"Pose  zeros: {zero_pct(pose):.1f}% of frames")
    print(f"Face  zeros: {zero_pct(face):.1f}% of frames")
    print(f"Left  hand zeros: {zero_pct(lh):.1f}% of frames")
    print(f"Right hand zeros: {zero_pct(rh):.1f}% of frames")
    print(f"Coordinate range: [{data.min():.3f}, {data.max():.3f}]")


def visualize_sample(sample_path, frame_idx=0):
    data = np.load(sample_path)
    n_frames = len(data)
    print(f"Visualizing {sample_path.name} — {n_frames} frames. Press any key to advance, Q to quit.")

    indices = [frame_idx] if frame_idx else range(n_frames)

    for i in indices:
        frame = data[i]
        pose = frame[:POSE_END].reshape(33, 4)
        face = frame[POSE_END:FACE_END].reshape(-1, 3)
        lh   = frame[FACE_END:LH_END].reshape(21, 3)
        rh   = frame[LH_END:RH_END].reshape(21, 3)

        canvas = np.zeros((480, 848, 3), dtype=np.uint8)

        def draw(landmarks, color):
            for lm in landmarks:
                if not np.all(lm[:2] == 0):  # skip zero landmarks
                    x = int(lm[0] * 848)
                    y = int(lm[1] * 480)
                    if 0 <= x < 848 and 0 <= y < 480:
                        cv2.circle(canvas, (x, y), 4, color, -1)

        draw(pose[:, :3], (0, 255, 0))   # green  — pose
        draw(face,        (255, 255, 0)) # yellow — face
        draw(lh,          (255, 80, 80)) # blue   — left hand
        draw(rh,          (80, 80, 255)) # red    — right hand

        cv2.putText(canvas, f"Frame {i}/{n_frames-1}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.imshow("Sanity Check", canvas)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


def animate_sample(sample_path, fps=30):
    data = np.load(sample_path)
    delay = int(1000 / fps)

    for i, frame in enumerate(data):
        pose = frame[:POSE_END].reshape(33, 4)
        face = frame[POSE_END:FACE_END].reshape(-1, 3)
        lh   = frame[FACE_END:LH_END].reshape(21, 3)
        rh   = frame[LH_END:RH_END].reshape(21, 3)

        canvas = np.zeros((480, 848, 3), dtype=np.uint8)

        # Use this before normalization

        # def draw(landmarks, color):
        #     for lm in landmarks:
        #         if not np.all(lm[:2] == 0):
        #             x, y = int(lm[0] * 848), int(lm[1] * 480)
        #             if 0 <= x < 848 and 0 <= y < 480:
        #                 cv2.circle(canvas, (x, y), 4, color, -1)


        # after normalization (landmarks are centered around shoulders and scaled, so we can just add an offset to visualize)
        def draw(landmarks, color):
            for lm in landmarks:
                if not np.all(lm[:2] == 0):
                    # remap from normalized space to screen
                    x = int((lm[0] + 2) / 4 * 848)  # assumes range [-2, 2]
                    y = int((lm[1] + 2) / 4 * 480)
                    if 0 <= x < 848 and 0 <= y < 480:
                        cv2.circle(canvas, (x, y), 4, color, -1)

        draw(pose[:, :3], (0, 255, 0))
        draw(face,        (255, 255, 0))
        draw(lh,          (255, 80, 80))
        draw(rh,          (80, 80, 255))

        cv2.putText(canvas, f"Frame {i}/{len(data)-1}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.imshow("Animation", canvas)
        if cv2.waitKey(delay) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    dataset_stats()

    sample = PROCESSED_DATA_DIR / "train/signer01/s01_g002_r0_rgb.npy"
    sample_stats(sample)
    animate_sample(sample) 
    sample = PROCESSED_DATA_DIR / "train/signer01/s01_g002_r1_rgb.npy"
    sample_stats(sample)
    animate_sample(sample)
    sample = PROCESSED_DATA_DIR / "train/signer01/s01_g003_r2_rgb.npy"
    sample_stats(sample)
    animate_sample(sample)
    sample = PROCESSED_DATA_DIR / "train/signer01/s01_g046_r2_rgb.npy"
    sample_stats(sample)
    animate_sample(sample)
