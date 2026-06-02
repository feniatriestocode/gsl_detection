import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch

from gsl_detect.config import CSV, PROCESSED_DATA_DIR
from gsl_detect.features import GSLLandmarker
from gsl_detect.modeling.model import GSLTransformer
from gsl_detect.modeling.dataset import GSLDataset
from gsl_detect.normalization import pad_or_truncate, pr_normalize_sequence

@st.cache_resource
def initialize_pipeline():
    """Initializes dataset properties and builds the model structure securely once."""
    reference_dataset = GSLDataset(CSV, PROCESSED_DATA_DIR / "train")
    dynamic_num_classes = reference_dataset.num_classes
    
    model = GSLTransformer(
        input_dim=318, 
        d_model=256, 
        nhead=8, 
        num_layers=4, 
        num_classes=dynamic_num_classes
    )
    
    model_path = Path("models/best_model.pt")
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model, reference_dataset

model, reference_dataset = initialize_pipeline()
landmarker = GSLLandmarker()

# --- Tab Layout Creation ---
tab1, tab2 = st.tabs(["📁 Upload Video Clip", "📹 Live Continuous Stream"])

# === TAB 1: FILE OVERVIEW PIPELINE ===
with tab1:
    st.subheader("Evaluate Recorded Sequences")
    uploaded_file = st.file_uploader("Select a GSL phrase video record...", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        st.video(uploaded_file)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_file.read())
            temp_path = tfile.name

        with st.spinner("Processing continuous frames and mapping meshes..."):
            raw_sequence = landmarker.process_video(temp_path)
            
            if raw_sequence is not None and len(raw_sequence) > 0:
                normalized_seq = pr_normalize_sequence(raw_sequence)
                padded_seq, mask = pad_or_truncate(normalized_seq)
                
                X_tensor = torch.tensor(padded_seq, dtype=torch.float32).unsqueeze(0)
                mask_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
                
                with torch.no_grad():
                    logits = model(X_tensor, mask_tensor)
                    prediction_idx = torch.argmax(logits, dim=1).item()
                    
                    gloss_map = list(reference_dataset.label_encoder.classes_)

                    if prediction_idx < len(gloss_map):
                        predicted_word = gloss_map[prediction_idx]
                    else:
                        predicted_word = f"Unknown Index {prediction_idx}"
                st.success(f"Predicted Gloss Translation: **{predicted_word}**")
            else:
                st.error("No skeletal landmark meshes detected in this video file. Ensure proper lighting.")


# === TAB 2: LIVE CAMERA FEED CONTINUOUS LOOP ===
# === TAB 2: LIVE CAMERA FEED WITH LANDMARK DRAWING ===
with tab2:
    st.subheader("Live Continuous Feed Translation")
    st.write("Sign directly into your camera. The system accumulates sequential frame coordinates over time to predict signs.")

    run_stream = st.checkbox("Power Camera Connection", key="stream_active_toggle")
    
    FRAME_WINDOW = st.empty()
    TEXT_WINDOW = st.empty()
    
    if run_stream:
        # Check alternative indexes based on your previous fix
        camera = cv2.VideoCapture(1, cv2.CAP_V4L2)
        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
            
        if not camera.isOpened():
            st.error("❌ Could not connect to webcam hardware.")
            run_stream = False
        else:
            TARGET_SEQUENCE_LENGTH = 60 
            live_sequence_buffer = []
            TEXT_WINDOW.info("📹 Camera connected. Drawing skeletal tracking structures...")
            
            # --- INITIALIZE MEDIAPIPE DRAWING UTILS ---
            import mediapipe as mp
            from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
            from mediapipe.tasks.python.vision import drawing_styles as mp_drawing_styles
            from mediapipe.tasks import python
            mp_drawing =mp_drawing

            # 4. Draw Face landmarks on the display matrix frame
            # if face_res.face_landmarks:
            #     for face_landmarks in face_res.face_landmarks:
            #         mp_drawing.draw_landmarks(
            #             image=rgb_frame,
            #             landmark_list=face_landmarks,
            #             # Connects the points to display facial expression grids
            #             connections=mp.tasks.vision.FaceLandmarksConnections.FACEMESH_TESSELLATION,
            #             landmark_drawing_spec=None, # Uses standard point markers
            #             connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            #         )

            while run_stream:
                success, frame = camera.read()
                if not success:
                    st.error("Hardware video pipeline interrupted.")
                    break
                
                # Convert BGR frame from OpenCV to RGB color mapping
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                try:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    
                    # 1. Run standard landmark detectors
                    pose_res = landmarker.pose.detect_for_video(mp_image, 33)
                    hand_res = landmarker.hands.detect_for_video(mp_image, 33)
                    face_res = landmarker.face.detect_for_video(mp_image, 33)
                    
                    # 2. Extract numerical values for your Transformer buffer
                    frame_packed = landmarker._pack_landmarks(pose_res, hand_res, face_res)
                    live_sequence_buffer.append(frame_packed)
                    
                    # 3. ─── NEW DRAWING LAYER BLOCK ─────────────────────────────────
                    # We draw directly onto the rgb_frame matrix that gets painted on the screen
                    
                    # Draw Pose skeleton connections
                    if pose_res.pose_landmarks:
                        # MediaPipe Tasks outputs need a slight conversion or matching drawer depending on version.
                        # If your landmarker uses standard MediaPipe Solutions, this draws it instantly:
                        for landmarks in pose_res.pose_landmarks:
                            mp_drawing.draw_landmarks(
                                rgb_frame, landmarks, mp_holistic.POSE_CONNECTIONS
                            )
                            
                    # Draw Hand skeleton connections (Left & Right)
                    if hand_res.hand_landmarks:
                        for landmarks in hand_res.hand_landmarks:
                            mp_drawing.draw_landmarks(
                                rgb_frame, landmarks, mp_holistic.HAND_CONNECTIONS
                            )
                except Exception as e:
                    # Catch frame exceptions quietly without breaking the continuous capture loop
                    pass

                # Render the frame (now complete with drawn landmark annotations!)
                FRAME_WINDOW.image(rgb_frame)

                # 4. Predict whenever sequence accumulates target length
                if len(live_sequence_buffer) >= TARGET_SEQUENCE_LENGTH:
                    raw_sequence = np.array(live_sequence_buffer)
                    normalized_seq = pr_normalize_sequence(raw_sequence)
                    padded_seq, mask = pad_or_truncate(normalized_seq)
                    
                    X_tensor = torch.tensor(padded_seq, dtype=torch.float32).unsqueeze(0)
                    mask_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
                    
                    with torch.no_grad():
                        logits = model(X_tensor, mask_tensor)
                        prediction_idx = torch.argmax(logits, dim=1).item()
                        gloss_map = list(reference_dataset.label_encoder.classes_)

                        if prediction_idx < len(gloss_map):
                            predicted_word = gloss_map[prediction_idx]
                        else:
                            predicted_word = f"Unknown Index {prediction_idx}"
                    
                    TEXT_WINDOW.success(f"Predicted GSL Sign: **{predicted_word}**")
                    live_sequence_buffer.clear()
            
            camera.release()
            FRAME_WINDOW.empty()
            TEXT_WINDOW.write("Camera Standby.")
    else:
        FRAME_WINDOW.write("Camera Disconnected.")
        