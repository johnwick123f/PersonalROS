## First you must install LLAMA.cpp with cublas
!git clone https://github.com/ggerganov/llama.cpp
%cd llama.cpp
!make LLAMA_CUBLAS=1

### Then download the model, the following is mistral llava 1.6 and its nice quality + good speed.
!wget https://huggingface.co/cmp-nct/llava-1.6-gguf/resolve/main/ggml-mistral-q_4_k.gguf

# Run the server for llama.cpp, Important to do nohup and & as that will execute it in background, so its possible to run other things
!nohup ./server -m "/content/drive/MyDrive/mistral-7b-q_5_k.gguf" -c 2048 -ngl 99 --mmproj "/content/drive/MyDrive/mmproj-mistral7b-f16.gguf" --host 0.0.0.0 --port 8000 &

# imports needed
import requests
import base64
import json
# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
    
def llava_gen(prompt, image):
  base64_image = encode_image("/content/fullbblock.jpg")
  headers = {
      'Content-Type': 'application/json',
  }

  json_data = {
    'image_data': [{
        'data': base64_image, 
        'id': 10
    }],
    'prompt': 'USER:[img-10]Describe the image.\nASSISTANT:',
    'stream': True
  }

  print()
  response = requests.post("http://0.0.0.0:8000/completion", headers=headers, json=json_data, stream=True)
  for chunk in response.iter_content(chunk_size=128):
      content = chunk.decode().strip().split('\n\n')[0]
      try:
          content_split = content.split('data: ')
          if len(content_split) > 1:
              content_json = json.loads(content_split[1])
              print(content_json["content"], end='', flush=True)
      except json.JSONDecodeError:
          break
