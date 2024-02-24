# Simple speech generation using piper tts(super fast and nice speed)

# stuff needed to install
!pip3 install onnxruntime-gpu
!pip install piper-tts

# Very simple function for speech generation
def SpeechGen(text):
    !echo f"{text}" | piper \
      --model en_US-lessac-medium \
      --output_file out.wav
