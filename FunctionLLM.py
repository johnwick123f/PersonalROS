#Should install llama-cpp-python first with CUBLAS
# Set gpu layers to -1, and the best model that can do all tasks i found is DPOPENHERMES V2 7b.
#Install cpp llama !CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
#Download model !wget https://huggingface.co/TheBloke/DPOpenHermes-7B-v2-GGUF/resolve/main/dpopenhermes-7b-v2.Q4_K_M.gguf

#Import stuff and Load model
from llama_cpp import Llama
llm = Llama(
      model_path="/content/dpopenhermes-7b-v2.Q4_K_M.gguf",
      n_gpu_layers=-1
)

# Down is the important prompt for functions
prompt = f'''You can only use these functions to solve the task
robot(task): Use this function to make a robot do some task. Can automatically detect objects.
VisualQA(task): Use this function to send some question from the task to a Visual answering model.
AudioQA:(task): Only Use this function for audio related questions
VisualChat(task): Use this function for more detailed and better explanation, Not good for simple few word answers.

Remember to be smart, high quality, and use common sense.
The task is, What color is the bottle, and grab it.<|im_end|>
<|im_start|>assistant
# Step 1: Determine its color using VisualQA
VisualQA("What is the color of the bottle?")
# Step 2: Robotic action
robot(f"grab the bottle")<|im_end|>
<|im_start|>user
# Please understand that you should only output the functions told.
Good job!, Now the next task is: Roughly explain the environment please!, Then can you grab the red thing<|im_end|>
<|im_start|>assistant
# Step 1: Describe the environment using VisualChat
VisualChat("Please describe the environment.")
# Step 2: Robotic action
robot("grab the red object")<|im_end|>
<|im_start|>user
# Assume that the robot function can detect objects automatically.
Perfect, now the task is Put the blue block on the green block<|im_end|>
<|im_start|>assistant
robot("put the blue block on the green block")<|im_end|>
<|im_start|>user
{prompt}'''

# Simple function for Generating
def function_llm(prompt):
  messages = [
          {"role": "system", "content": "You are a intelligent, super smart, high quality ai that uses functions."},
          {
              "role": "user",
              "content": f"{prompt}"
          }
      ]
  for token in llm.create_chat_completion(messages = messages,stream=True):
    out = token["choices"][0]["delta"]
    has_choices = "role" in out
    if not has_choices:
      ok = "content" in out
      if ok: 
        yield (out["content"])
      else: pass
