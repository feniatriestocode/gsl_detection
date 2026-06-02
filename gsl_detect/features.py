import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from multiprocessing import Pool
import os
from gsl_detect.config import POSE_SELECTED_INDICES, GLOSSES, RAW_DATA_DIR, INTERIM_DATA_DIR

from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions,
    HandLandmarker, HandLandmarkerOptions
)

BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode


class GSLLandmarker:
    def __init__(self):
        self._timestamp_ms = 0

        self.pose = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
                running_mode=RunningMode.VIDEO,
                min_pose_detection_confidence=0.5
            )
        )

        self.hands = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
                running_mode=RunningMode.VIDEO,
                num_hands=2
            )
        )

    def process_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        sequence = []
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            self._timestamp_ms += 33
            pose_res = self.pose.detect_for_video(mp_image, self._timestamp_ms)
            hand_res = self.hands.detect_for_video(mp_image, self._timestamp_ms)

            sequence.append(self._pack_landmarks(pose_res, hand_res))

        cap.release()
        return np.array(sequence) if sequence else None

    def _pack_landmarks(self, pose_result, hand_result):
        # Pose: 15 pts × 4 coords (x, y, z, visibility) = 60 features
        pose = np.array([[pose_result.pose_landmarks[0][i].x,
                          pose_result.pose_landmarks[0][i].y,
                          pose_result.pose_landmarks[0][i].z,
                          pose_result.pose_landmarks[0][i].visibility]
                         for i in POSE_SELECTED_INDICES]).flatten() \
               if pose_result.pose_landmarks else np.zeros(len(POSE_SELECTED_INDICES) * 4)

        # Hands: 21 pts × 3 coords = 63 features each
        lh = np.zeros(21 * 3)
        rh = np.zeros(21 * 3)
        for i, handedness in enumerate(hand_result.handedness):
            label = handedness[0].category_name  # 'Left' or 'Right'
            coords = np.array([[lm.x, lm.y, lm.z]
                                for lm in hand_result.hand_landmarks[i]]).flatten()
            if label == 'Left':
                lh = coords
            else:
                rh = coords

        # Total: 60 + 63 + 63 = 186 features
        return np.concatenate([pose, lh, rh])


def generate_features():
    csv_path = RAW_DATA_DIR / "isolated_GSL_corpus.csv"
    df = pd.read_csv(csv_path)
    df["Signer"] = df["Video"].apply(lambda x: x.split("/")[0])

    all_signers = sorted(df["Signer"].unique())  # 21 signers

    signer_groups = [all_signers[i::4] for i in range(4)]

    tasks = []
    for group in signer_groups:
        group_videos = df[df["Signer"].isin(group)]["Video"].tolist()
        tasks.append(group_videos)

    with Pool(processes=8) as pool:
        pool.map(process_group, tasks)


def process_group(video_list):
    """Each process handles its own list of videos sequentially."""
    landmarker = GSLLandmarker()  # one instance per process

    for video_rel_path in tqdm(video_list, position=0):
        video_file = GLOSSES / f"{video_rel_path}.mp4"
        if not video_file.exists():
            continue

        data_sequence = landmarker.process_video(video_file)

        if data_sequence is not None:
            output_file = INTERIM_DATA_DIR / f"{video_rel_path}_proc.npy"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_file, data_sequence)


if __name__ == "__main__":
    generate_features()
