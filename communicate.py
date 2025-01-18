import pyttsx3

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
        pass  # Ignore the error if the loop is already running