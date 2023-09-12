import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import mediapipe as mp
from xgboost import XGBClassifier
import numpy as np
import av
import cv2
from streamlit_webrtc import webrtc_streamer
# initializing mediapipe
mpHands = mp.solutions.hands    # this performs the hand recognition
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)    # this line configures the model
mpDraw = mp.solutions.drawing_utils  # this line draws the detected keypoints


def video_frame_callback(frame):
    handpoints = ['HandLandmark.WRIST_lmx', 'HandLandmark.WRIST_lmy', 'HandLandmark.THUMB_CMC_lmx', 'HandLandmark.THUMB_CMC_lmy', 'HandLandmark.THUMB_MCP_lmx', 'HandLandmark.THUMB_MCP_lmy', 'HandLandmark.THUMB_IP_lmx', 'HandLandmark.THUMB_IP_lmy', 'HandLandmark.THUMB_TIP_lmx', 'HandLandmark.THUMB_TIP_lmy', 'HandLandmark.INDEX_FINGER_MCP_lmx', 'HandLandmark.INDEX_FINGER_MCP_lmy', 'HandLandmark.INDEX_FINGER_PIP_lmx', 'HandLandmark.INDEX_FINGER_PIP_lmy', 'HandLandmark.INDEX_FINGER_DIP_lmx', 'HandLandmark.INDEX_FINGER_DIP_lmy', 'HandLandmark.INDEX_FINGER_TIP_lmx', 'HandLandmark.INDEX_FINGER_TIP_lmy', 'HandLandmark.MIDDLE_FINGER_MCP_lmx', 'HandLandmark.MIDDLE_FINGER_MCP_lmy', 'HandLandmark.MIDDLE_FINGER_PIP_lmx',
              'HandLandmark.MIDDLE_FINGER_PIP_lmy', 'HandLandmark.MIDDLE_FINGER_DIP_lmx', 'HandLandmark.MIDDLE_FINGER_DIP_lmy', 'HandLandmark.MIDDLE_FINGER_TIP_lmx', 'HandLandmark.MIDDLE_FINGER_TIP_lmy', 'HandLandmark.RING_FINGER_MCP_lmx', 'HandLandmark.RING_FINGER_MCP_lmy', 'HandLandmark.RING_FINGER_PIP_lmx', 'HandLandmark.RING_FINGER_PIP_lmy', 'HandLandmark.RING_FINGER_DIP_lmx', 'HandLandmark.RING_FINGER_DIP_lmy', 'HandLandmark.RING_FINGER_TIP_lmx', 'HandLandmark.RING_FINGER_TIP_lmy', 'HandLandmark.PINKY_MCP_lmx', 'HandLandmark.PINKY_MCP_lmy', 'HandLandmark.PINKY_PIP_lmx', 'HandLandmark.PINKY_PIP_lmy', 'HandLandmark.PINKY_DIP_lmx', 'HandLandmark.PINKY_DIP_lmy', 'HandLandmark.PINKY_TIP_lmx', 'HandLandmark.PINKY_TIP_lmy']

    all_gestures = []
    img = frame.to_ndarray(format="bgr24")





    
    flipped = img[::-1,:,:]
    frame_rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    print(result)










    
    try:
        x, y, c = img.shape
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for handslms in result.multi_hand_landmarks:
                lmks = []
                for lm in handslms.landmark:
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    lmks.extend([lmx, lmy])

                preds = model_xgb.predict(np.array(lmks).reshape(1, -1))
                predicted_names = [k for k, v in gesture_names.items() if v == preds]
                placeholder.header(f"Do you mean: :green[{str(predicted_names[0])}]?")

                gesture = dict(zip(handpoints, lmks))
                all_gestures.append(gesture)

                mpDraw.draw_landmarks(img, handslms, mpHands.HAND_CONNECTIONS,
                                      mpDraw.DrawingSpec(color=(3, 252, 244), thickness=2, circle_radius=2),
                                      mpDraw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
    except Exception as e:
        st.error(f"Error processing video frame: {e}")
    return av.VideoFrame.from_ndarray(img, format="bgr24")

model_xgb = XGBClassifier()
model_xgb.load_model("model.json")
gesture_names = {'A': 0,
 'B': 1,
 'C': 2,
 'D': 3,
 'E': 4,
 'F': 5,
 'G': 6,
 'H': 7,
 'I': 8,
 'J': 9,
 'K': 10,
 'L': 11,
 'M': 12,
 'N': 13,
 'O': 14,
 'P': 15,
 'Q': 16,
 'R': 17,
 'S': 18,
 'T': 19,
 'U': 20,
 'V': 21,
 'W': 22,
 'X': 23,
 'Y': 24,
 'Z': 25}
st.set_page_config(page_title="PRACTICE: The American Sign Language (ASL) sign",
                   page_icon="🧏🏼",
                   layout="wide",)
st.title("Practice your ASL")
st.image('./ASL.png', width = 120)
st.success("American Sign Language (ASL) is a natural language that serves as "
         "the predominant sign language of Deaf communities in the United States "
         "of America and most of Anglophone Canada. ASL is a complete and "
         "organized visual language that is expressed by employing both manual "
         "and nonmanual features.")
st.warning("All video streaming will not be stored anywhere. Feel free to try it out. ")
left_column, right_column = st.columns(2)
with left_column:
    FRAME_WINDOW = st.image([])
    webrtc_streamer(key="example", video_frame_callback=video_frame_callback, media_stream_constraints={"video": True, "audio": False}, async_processing=True,)
    placeholder = st.empty()
 
with right_column:
    ASL_poster = st.image('./ASL_poster.jpg', width = 420)

# initializing webcam for video capture
#cap = cv2.VideoCapture(0)
#webrtc_ctx = webrtc_streamer(
#    key="video_in",
#    media_stream_constraints={"video": True, "audio": False},
#)



