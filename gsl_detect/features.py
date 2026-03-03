import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from pathlib import Path
from tqdm import tqdm 
from loguru import logger

from gsl_detect.config import FACE_SELECTED_INDICES, GLOSSES, RAW_DATA_DIR, PROCESSED_DATA_DIR

from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions,
    FaceLandmarker, FaceLandmarkerOptions,
    HandLandmarker, HandLandmarkerOptions
)

from mediapipe.tasks.python.components.containers import NormalizedLandmark

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

        self.face = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path='face_landmarker.task'),
                running_mode=RunningMode.VIDEO,
                min_face_detection_confidence=0.5
            )
        )

        self.hands = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
                running_mode=RunningMode.VIDEO
            )
        )

    def process_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        sequence = []
        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            self._timestamp_ms += 33            
            pose_res = self.pose.detect_for_video(mp_image, self._timestamp_ms)
            hand_res = self.hands.detect_for_video(mp_image, self._timestamp_ms)
            face_res = self.face.detect_for_video(mp_image, self._timestamp_ms)

            sequence.append(self._pack_landmarks(pose_res, hand_res, face_res))

        cap.release()
        return np.array(sequence) if sequence else None

    def _pack_landmarks(self, pose_result, hand_result, face_result):
        # Pose: 33 pts * 4 values = 132
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                         for lm in pose_result.pose_landmarks[0]]).flatten() \
               if pose_result.pose_landmarks else np.zeros(33 * 4)

        # Face: selected indices * 3 values
        face = np.array([[face_result.face_landmarks[0][i].x,
                          face_result.face_landmarks[0][i].y,
                          face_result.face_landmarks[0][i].z]
                         for i in FACE_SELECTED_INDICES]).flatten() \
               if face_result.face_landmarks else np.zeros(len(FACE_SELECTED_INDICES) * 3)

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

        return np.concatenate([pose, face, lh, rh])

def generate_features():
    """
    Main loop to process the GSL dataset based on your CSV and folder structure.
    """
    logger.info("Initializing Landmarker Task...")
    try:
        landmarker = GSLLandmarker()
    except Exception as e:
        logger.error(f"Failed to load Landmarker: {e}")
        return

    csv_path = RAW_DATA_DIR / "isolated_GSL_corpus.csv" 
    if not csv_path.exists():
        logger.error(f"CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting extraction for {len(df)} samples...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)): 
        video_rel_path = row.get('Video', row.get('video_path'))
        video_file = GLOSSES / f"{video_rel_path}.mp4"

        if not video_file.exists():
            logger.warning(f"Video file not found: {video_file}")
            continue
            
        data_sequence = landmarker.process_video(video_file)
        
        if data_sequence is not None:
            # Save as sample_00001.npy for consistency
            output_file = PROCESSED_DATA_DIR / f"{video_rel_path}_proc.npy"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_file, data_sequence)

    logger.success("Preprocessing complete!")

if __name__ == "__main__":
    generate_features()