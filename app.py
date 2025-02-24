__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os
from pathlib import Path
import time
import base64
import rag_agent
import speech_to_text
import translation
import text_to_speech
import scipy.io.wavfile as wav
import numpy as np
from pygame import mixer
import pygame
import voice_module
from threading import Lock
import base64

audio_buffer = []
audio_buffer_lock = Lock()

st.set_page_config(layout = "wide")

class FileManager:
    def __init__(self):
        # Create directory in the Streamlit runtime
        self.base_dir = Path(tempfile.gettempdir()) / "pdf_voice_chat"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Fixed filenames
        self.pdf_path = self.base_dir / "current.pdf"
        self.user_audio_path = self.base_dir / "user_audio.wav"
        self.bot_audio_path = self.base_dir / "bot_audio.wav"
    
    def save_pdf(self, uploaded_file):
        """Save uploaded PDF, overwriting if exists."""
        if uploaded_file is not None:
            with open(self.pdf_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            return str(self.pdf_path)
        return None
    
    def save_user_audio(self, audio_data, sample_rate):
        """Save recorded audio, overwriting if exists."""
        sf.write(str(self.user_audio_path), audio_data, sample_rate)
        return str(self.user_audio_path)
    
    def save_bot_audio(self, audio_bytes):
        """Save bot's audio response with verification."""
        print("Saving audio path")
        try:
            # Convert BytesIO to bytes if necessary
            if hasattr(audio_bytes, 'getvalue'):
                audio_bytes = audio_bytes.getvalue()
                
            with open(self.bot_audio_path, "wb") as f:
                f.write(audio_bytes)
            print("Audio file writing complete")
            # Verify file was written
            if self.bot_audio_path.exists():
                print(f"Bot audio saved successfully. Size: {self.bot_audio_path.stat().st_size} bytes")
            return str(self.bot_audio_path)
        except Exception as e:
            print(f"Error saving bot audio: {str(e)}")
            return None

def autoplay_audio(audio_path: str):
    """Autoplay audio in Streamlit with error handling."""
    st.audio(audio_path)
    # try:
    #     with open(audio_path, "rb") as f:
    #         audio_bytes = f.read()
        
    #     # Add debug print to check audio file size
    #     print(f"Audio file size: {len(audio_bytes)} bytes")
        
    #     audio_base64 = base64.b64encode(audio_bytes).decode()
    #     # Modified audio tag with controls for debugging
    #     audio_tag = f'''
    #         <audio autoplay="true" controls="true" src="data:audio/wav;base64,{audio_base64}">
    #         Your browser does not support the audio element.
    #         </audio>
    #     '''
    #     st.markdown(audio_tag, unsafe_allow_html=True)
    # except Exception as e:
    #     st.error(f"Error playing audio: {str(e)}")
    #     print(f"Audio playback error: {str(e)}")

# Initialize ALL session state variables at the start
if 'file_manager' not in st.session_state:
    st.session_state.file_manager = FileManager()
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'is_bot_speaking' not in st.session_state:
    st.session_state.is_bot_speaking = False
if 'recording_started' not in st.session_state:
    st.session_state.recording_started = False
if 'stream' not in st.session_state:
    st.session_state.stream = None
if 'audio_data' not in st.session_state:  # Make sure this is initialized
    st.session_state.audio_data = list()
if 'rag_chain' not in st.session_state:  # Add this to persist the RAG chain
    st.session_state.rag_chain = None
if 'chat_history' not in st.session_state:  # Add this to persist chat history
    st.session_state.chat_history = []

# Styling
st.markdown("""
    <style>
    .chat-container {
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background-color: #f0f2f6;
    }
    .recording-animation {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background-color: red;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.2); opacity: 0.8; }
        100% { transform: scale(1); opacity: 1; }
    }
    .bot-speaking-animation {
        display: flex;
        gap: 5px;
        align-items: center;
    }
    .bot-speaking-bar {
        width: 4px;
        height: 20px;
        background-color: #2E86C1;
        animation: soundBars 0.5s infinite;
    }
    @keyframes soundBars {
        0% { height: 20px; }
        50% { height: 40px; }
        100% { height: 20px; }
    }
    </style>
""", unsafe_allow_html=True)

def start_recording():
    """Start recording audio."""
    global audio_buffer
    audio_buffer = []  # Reset the buffer
    
    def callback(indata, frames, time, status):
        if status:
            print(status)
        with audio_buffer_lock:
            audio_buffer.append(indata.copy())
    
    # Start the recording stream
    stream = sd.InputStream(callback=callback,
                          channels=1,
                          samplerate=44100,
                          dtype='float32')
    return stream

def stop_recording(stream):
    """Stop recording and return the audio data."""
    stream.stop()
    stream.close()
    
    # Combine all audio chunks
    if audio_buffer_lock:
        print("Stopped Recording. Audio buffer:", audio_buffer)
        audio_buffer_np = np.array(audio_buffer)
        combined_audio = np.concatenate(audio_buffer_np, axis=0)
        return combined_audio, 44100
    return None, None

def create_bot_speaking_animation():
    """Create animated bars to indicate bot is speaking."""
    cols = st.columns(5)
    for i, col in enumerate(cols):
        col.markdown(
            f"""<div class="bot-speaking-bar" 
            style="animation-delay: {i * 0.1}s"></div>""",
            unsafe_allow_html=True
        )

def record_audio(silence_threshold=0.005, silence_duration=5, sample_rate=44100, recording_file="user_recording.wav"):
    """
    Record audio from the microphone until silence is detected
    silence_threshold: amplitude threshold below which audio is considered silence
    silence_duration: duration of silence (in seconds) before stopping
    sample_rate: sample rate in Hz
    """
    # Create user_input directory if it doesn't exist
    if not os.path.exists('user_input'):
        os.makedirs('user_input')
    
    print("Recording... (speak now, will stop after 5 seconds of silence)")
    
    # Initialize variables for silence detection
    silence_samples = int(silence_duration * sample_rate)
    buffer = []
    silent_frames = 0
    
    def callback(indata, frames, time, status):
        nonlocal silent_frames
        if status:
            print(status)
        buffer.extend(indata[:, 0])  # Only take first channel
        # Check if the latest frame is silent
        if np.abs(indata).mean() < silence_threshold:
            silent_frames += len(indata)
        else:
            silent_frames = 0
    
    # Start recording
    with sd.InputStream(samplerate=sample_rate, 
                       channels=1, 
                       callback=callback):
        while silent_frames < silence_samples:
            sd.sleep(100)  # Sleep for 100ms
    
    print("Recording finished!")
    
    # Convert buffer to numpy array
    recording = np.array(buffer)
    
    # Save as WAV
    user_audio_path = st.session_state.file_manager.save_user_audio(
                            recording, 
                            sample_rate
                        )
    
    return user_audio_path

def show_pdf(file):
    # Display PDF in an iframe
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    st.title("Multi-lingual PDF Voice Chat")
    
    # PDF Upload Section
    st.header("1. Upload Your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file:
        st.session_state.pdf_path = st.session_state.file_manager.save_pdf(uploaded_file)
        st.success(f"PDF uploaded successfully: {uploaded_file.name}")
        # st.set_page_config(page_title="PDF Viewer", layout="wide")
        show_pdf(uploaded_file)

        # rag_chain = rag_agent.setup_pdf_rag(filename=st.session_state.pdf_path)
        # chat_history = []

    # Voice Chat Section
    st.header("2. Voice Chat")
    
    if st.session_state.pdf_path:
        if 'rag_chain' not in st.session_state or st.session_state.rag_chain is None:
            st.session_state.rag_chain = rag_agent.setup_pdf_rag(filename=st.session_state.pdf_path)
        
        col1, col2 = st.columns(2)
        
        # Recording controls
        with col1:
            if st.button("Ask me Anything"):
                st.markdown("""
                    <div style=''background-color: #E8F0FE; padding: 20px; border-radius: 10px'>
                        <p style='color: #424242;'>Recording... (speak now, will stop after 5 seconds of silence).</p>
                    </div>
                """, unsafe_allow_html=True)
                st.session_state.is_recording = True
                user_audio_path = record_audio()
                st.markdown("""
                    <div style=''background-color: #E8F0FE; padding: 20px; border-radius: 10px'>
                        <p style='color: #424242;'>Recording complete.</p>
                    </div>
                """, unsafe_allow_html=True)
                extracted_text, extracted_lang = voice_module.process_query(audio_file=user_audio_path)
                print(extracted_text, extracted_lang)
                st.markdown(f"""
                    <div style=''background-color: #E8F0FE; padding: 20px; border-radius: 10px'>
                        <p style='color: #424242;'>Extracted and converted your question into English text</p>
                    </div>
                """, unsafe_allow_html=True)
                rag_response = rag_agent.rag_agent_response({
                    "input": f"{extracted_text}"},
                    st.session_state.rag_chain,
                    st.session_state.chat_history
                )
                rag_response_text = rag_response["answer"]
                st.markdown(f"""
                    <div style=''background-color: #E8F0FE; padding: 20px; border-radius: 10px'>
                        <p style='color: #424242;'>Got RAG agent output.</p>
                    </div>
                """, unsafe_allow_html=True)
                st.session_state.chat_history.append((extracted_text, rag_response_text))
                translated_text = voice_module.process_rag_output(rag_response_text, extracted_lang)
                st.markdown(f"""
                    <div style=''background-color: #E8F0FE; padding: 20px; border-radius: 10px'>
                        <p style='color: #424242;'>Translated RAG Agent output</p>
                    </div>
                """, unsafe_allow_html=True)
                audio_stream = text_to_speech.text_to_speech_stream(translated_text, "pNInz6obpgDQGcFmaJgB")
                st.markdown("""
                    <div style=''background-color: #E8F0FE; padding: 20px; border-radius: 10px'>
                        <p style='color: #424242;'>Converted Text to Speech. Saving it to an audio file...</p>
                    </div>
                """, unsafe_allow_html=True)
                print("Converted text to speech")
                bot_response = audio_stream
                
                print("Saving bot response")
                st.markdown("""
                    <div style=''background-color: #E8F0FE; padding: 20px; border-radius: 10px'>
                        <p style='color: #424242;'>Play to listen to the bot's response. To ask a follow up, click the Ask Me Anything Button again.</p>
                    </div>
                """, unsafe_allow_html=True)
                bot_audio_path = st.session_state.file_manager.save_bot_audio(bot_response)
                st.markdown("<h3>Bot Response:</h3>", unsafe_allow_html=True)
                # create_bot_speaking_animation()
                autoplay_audio(bot_audio_path)
                
                # Reset recording state
                st.session_state.is_recording = False
                
            # Show recording animation when recording is in progress
            # if st.session_state.is_recording:
            #     st.markdown('<div class="recording-animation"></div>',
            #              unsafe_allow_html=True)
    else:
        st.warning("Please upload a PDF first to start the voice chat.")

if __name__ == "__main__":
    main()

