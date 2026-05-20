# Sign Language Recognition App with AI Assistant 

This Streamlit application processes a real-time webcam feed to recognize American Sign Language (ASL) gestures and accumulates them into text. It integrates a conversational AI assistant (Google Gemini) to help triage and communicate effectively based on the recognized text.

## Features
- **Real-Time Gesture Recognition**: Uses MediaPipe for hand tracking and a pre-trained Keras CNN model (`model.h5`) for ASL gesture prediction.
- **WebRTC Camera Stream**: Uses `streamlit-webrtc`, fully compatible with Streamlit Community Cloud and Hugging Face Spaces for zero-cost deployment.
- **AI Chatbot Integration**: Integrated Gemini 1.5 Flash assistant to contextually interpret your signed string and communicate naturally.

## Local Testing
1. Ensure your environment is set up and navigate into the `SignLanguageApp` directory:
   ```bash
   cd SignLanguageApp
   ```
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the application:
   ```bash
   streamlit run app.py
   ```0000

## Next Steps for Free Deployment
1. **Get a Free API Key**: Grab a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey) to power your chatbot.
2. **Test Locally**: Run the code (`streamlit run app.py`) to ensure the webcam captures your hand gestures smoothly.
3. **Deploy**: 
   - Push this `SignLanguageApp` folder to a free GitHub repository.
   - Go to [share.streamlit.io](https://share.streamlit.io), connect your GitHub account.
   - Select the repository, point it to `app.py`, and hit deploy. It will be live on the internet at zero cost.
