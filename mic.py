import speech_recognition as sr
def talk():
    # Initialize recognizer class (for recognizing the speech)
    recognizer = sr.Recognizer()

    #file_path = "micTestApple.wav" #reaplace with path to .wav file
#If using computer mic keep following with if not comment out and uncomment other block
    # Capture the audio from the microphone
    with sr.Microphone() as source:
        print("Say something!")
        audio = recognizer.listen(source)
    '''
     # Open the audio file as a source
    with sr.AudioFile(file_path) as source:
        print("Processing the audio file...")
        audio = recognizer.record(source)  # Read the entire audio file
    '''
    # Recognize the speech using Google Web Speech API
    try:
        print("You said: " + recognizer.recognize_google(audio))
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    return recognizer.recognize_google(audio)
