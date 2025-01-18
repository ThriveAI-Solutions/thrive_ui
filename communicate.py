import streamlit as st
import pyttsx3
import speech_recognition as sr

def listen():
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Use the microphone as the source for input
    with sr.Microphone() as source:

        # Adjust for ambient noise and record the audio
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            # Recognize speech using Google Web Speech API
            text = recognizer.recognize_google(audio)

            return text
        except sr.UnknownValueError:
            st.error('Sorry, I could not understand the audio.', icon="🚨")
            return None
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}", icon="🚨")
            return None

def speak(message):
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    # Set properties before adding text
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

    # Get available voices
    voices = engine.getProperty('voices')

    # Set the voice (0 for male, 1 for female, etc.)
    engine.setProperty('voice', voices[1].id)

    # Add the text to the queue
    engine.say(message)

    # Wait for the current speech to finish
    try:
        engine.runAndWait()
    except RuntimeError:
        st.error('Sorry, speech is already in progress.', icon="🚨")
        pass  # Ignore the error if the loop is already running
