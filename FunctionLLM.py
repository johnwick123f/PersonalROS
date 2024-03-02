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
system = f"""
You are a helpful assistant and your only goal is to use these functions.
{
  "name": "Robot",
  "description": "Completes some simple or complicated robot task. It has no memory however and can not describe things.",
  "parameters": {
    "type": "object",
    "properties": {
      "Task": {
        "type": "str"
      }
    }
  },
  "returns": "None"
}
{
  "name": "VisualQA",
  "description": "Get answers to any visual question and can describe images/scenes. not for manipulation",
  "parameters": {
    "type": "object",
    "properties": {
      "Question": {
        "type": "str"
      },
    }
  },
  "returns": "None"
}
{
  "name": "AudioQA",
  "description": "Answers any question about some audio"",
  "parameters": {
    "type": "object",
    "properties": {
      "Audio_path": {
        "type": "str"
      },
    }
  },
  "returns": "None"
}
To use these functions respond with: 
{"f": "function_name", "args": {"arg_1": "value_1", "arg_2": value_2, ...}}
{"f": "function_name", "args": {"arg_1": "value_1", "arg_2": value_2, ...}}

Do not use unneccesary functions but be sure to accurately and correctly complete the task
"""
prompt = ""
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
