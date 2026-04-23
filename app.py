import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import google.generativeai as genai
import av
import threading

# Explicit import for MediaPipe solutions to prevent AttributeError
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing

st.set_page_config(layout="wide", page_title="Sign Language Recognition 🤟")

@st.cache_resource
def load_slr_model():
    # Load the trained model from the repository
    return load_model("model.h5", compile=False)

try:
    model = load_slr_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    print(f"Error loading model: {e}")
    st.stop()

# Alphabet used by this model (A-Y, excluding J)
alpha = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']

class ThreadSafeString:
    def __init__(self):
        self.lock = threading.Lock()
        self.text = ""
        
    def add(self, char):
        with self.lock:
            self.text += char
            
    def get(self):
        with self.lock:
            return self.text
            
    def clear(self):
        with self.lock:
            self.text = ""

@st.cache_resource
def get_shared_state():
    return ThreadSafeString()

shared_state = get_shared_state()

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=1,  # Only track one hand to avoid confusion
            min_detection_confidence=0.5
        )
        self.last_predicted = ""
        self.frame_count = 0
        self.debounce_frames = 15

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Display the accumulated text natively on the video feed
        cv2.putText(img, f"Current text: {shared_state.get()}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, c = img.shape
                x_max, y_max = 0, 0
                x_min, y_min = w, h
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max: x_max = x
                    if x < x_min: x_min = x
                    if y > y_max: y_max = y
                    if y < y_min: y_min = y
                
                # Padding around the hand
                x_min = max(0, x_min - 20)
                y_min = max(0, y_min - 20)
                x_max = min(w, x_max + 20)
                y_max = min(h, y_max + 20)

                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                if x_max > x_min and y_max > y_min:
                    roi = img[y_min:y_max, x_min:x_max]
                    
                    if roi.size > 0:
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        blurred = cv2.GaussianBlur(gray, (15,15), 0)
                        resized_img = cv2.resize(blurred, (28,28))
                        
                        try:
                            # Bulletproof Model Prediction: 
                            # Tries CNN shape first, falls back to Dense shape if it fails
                            try:
                                data = np.asarray(resized_img.reshape(1, 28, 28, 1), dtype="float32") / 255.0
                                pred_probab = model.predict(data, verbose=0)[0]
                            except ValueError as e:
                                print(f"Shape error (trying fallback): {e}")
                                data = np.asarray(resized_img.reshape(1, 784), dtype="float32") / 255.0
                                pred_probab = model.predict(data, verbose=0)[0]
                                
                            pred_class = list(pred_probab).index(max(pred_probab))
                            confidence = max(pred_probab)
                            
                            char = alpha[pred_class]
                            
                            # Draw prediction on screen
                            cv2.putText(img, f"{char} ({confidence:.2f})", (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            
                            if confidence > 0.8:
                                if self.last_predicted == char:
                                    self.frame_count += 1
                                else:
                                    self.last_predicted = char
                                    self.frame_count = 1
                                
                                # Add character to string if holding for debounce_frames
                                if self.frame_count >= self.debounce_frames:
                                    shared_state.add(char)
                                    self.frame_count = 0 # Reset counter
                                    
                        except Exception as e:
                            print(f"Prediction Error: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.title("Sign Language Communication App 🤟💬")

col1, col2 = st.columns([1, 1], gap="large")

st.sidebar.title("Configuration")

if "GEMINI_API_KEY" in st.secrets:
    genai_api_key = st.secrets["GEMINI_API_KEY"]
else:
    genai_api_key = st.sidebar.text_input("Google AI Studio API Key", type="password", help="Get a free key from aistudio.google.com")

if genai_api_key:
    genai.configure(api_key=genai_api_key)
else:
    st.sidebar.warning("Please configure your Gemini API Key.")

debounce_val = st.sidebar.slider("Gesture Hold Time (Frames)", min_value=5, max_value=60, value=15)

with col1:
    st.header("📱 Camera Feed")
    st.markdown("Enable the webcam and start signing. Hold a gesture for a second to register it.")
    
    webrtc_ctx = webrtc_streamer(
        key="sign-lang",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        async_processing=True,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )
    
    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.debounce_frames = debounce_val
    
    st.markdown("---")
    st.subheader("Accumulated Text")
    
    text_container = st.container()
    col_a, col_b, col_c = st.columns(3)
    if col_a.button("🔄 Refresh View"):
        st.rerun()
    if col_b.button("␣ Add Space"):
        shared_state.add(" ")
        st.rerun()
    if col_c.button("❌ Clear"):
        shared_state.clear()
        st.rerun()
        
    text_container.info(f"**{shared_state.get()}**")

with col2:
    st.header("🤖 AI Communication Assistant")
    st.markdown("The assistant uses the sign language text you accumulate to help contextualize and communicate effectively.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if st.button("➤ Send Accumulated Text", use_container_width=True):
        recognized_text = shared_state.get()
        if not recognized_text:
            st.warning("No text accumulated yet from the camera.")
        elif not genai_api_key:
            st.warning("Please enter your Gemini API Key in the sidebar.")
        else:
            st.session_state.messages.append({"role": "user", "content": f"[From Sign Language camera]: {recognized_text}"})
            with st.chat_message("user"):
                st.markdown(f"*(Signed)*: {recognized_text}")
                
            shared_state.clear()
            
            with st.spinner("AI is thinking..."):
                try:
                    generation_config = {"temperature": 0.5, "top_p": 0.9, "max_output_tokens": 512}
                    gemini_model = genai.GenerativeModel(
                        "gemini-1.5-flash",
                        generation_config=generation_config,
                        system_instruction="You are a helpful communication and triage assistant. The user is communicating via a Sign Language Recognition camera. The text provided might be slightly misspelled or just a string of characters (e.g. 'HWLLO'). Interpret the user's intent, fix typos if obvious, and respond naturally to help them communicate their needs effectively.",
                    )
                    
                    history = []
                    for m in st.session_state.messages[:-1]:
                        role = "user" if m["role"] == "user" else "model"
                        history.append({"role": role, "parts": [m["content"]]})
                    
                    chat = gemini_model.start_chat(history=history)
                    response = chat.send_message(f"Please help me with this signed message: {recognized_text}")
                    
                    st.session_state.messages.append({"role": "model", "content": response.text})
                    with st.chat_message("model"):
                        st.markdown(response.text)
                except Exception as e:
                    st.error(f"Error communicating with Gemini API: {e}")
                    print(f"Gemini API Error: {e}")

    # Standard chat input as fallback
    user_input = st.chat_input("Type a message manually...")
    if user_input:
        if not genai_api_key:
            st.warning("Please enter your Gemini API Key in the sidebar.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
                
            with st.spinner("Wait..."):
                try:
                    gemini_model = genai.GenerativeModel(
                        "gemini-1.5-flash",
                        system_instruction="You are a helpful communication and triage assistant."
                    )
                    history = []
                    for m in st.session_state.messages[:-1]:
                        role = "user" if m["role"] == "user" else "model"
                        history.append({"role": role, "parts": [m["content"]]})
                    chat = gemini_model.start_chat(history=history)
                    response = chat.send_message(user_input)
                    st.session_state.messages.append({"role": "model", "content": response.text})
                    with st.chat_message("model"):
                        st.markdown(response.text)
                except Exception as e:
                    st.error(f"Error communicating with Gemini: {e}")
                    print(f"Gemini API Error (fallback): {e}")