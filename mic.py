import speech_recognition as sr
def talk():
    # Initialize recognizer class (for recognizing the speech)
    recognizer = sr.Recognizer()

    # Capture the audio from the microphone
    with sr.Microphone() as source:
        print("Say something!")
        audio = recognizer.listen(source)

    # Recognize the speech using Google Web Speech API
    try:
        print("You said: " + recognizer.recognize_google(audio))
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    return recognizer.recognize_google(audio)
