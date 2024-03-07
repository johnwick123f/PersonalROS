## Some functions for Generating with the llm
def function_llm(prompt, system):
  messages = [
          {"role": "system", "content": f"{system}"},
          {
              "role": "user",
              "content": f"{prompt}"
          }
      ]
  output = llm.create_chat_completion(messages = messages, stop=['\n'])
  return output["choices"][0]["message"]["content"]
def function_llm_streaming(prompt, system):
  messages = [
          {"role": "system", "content": f"{system}"},
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
      if ok: yield (out["content"])
      else: pass
## Utility functions
def format_prompt(new_prompt, old_prompt, response, num): ## simple function for formating prompts.
    if num == 0: return new_prompt
    formatted_list = []
    for i, prompt in enumerate(old_prompt):
        formatted = f"{prompt}<|im_end|>\n<|im_start|>assistant\n{response[i]}<|im_end|>\n<|im_start|>user"
        formatted_list.append(formatted)
    joined_sentence = " ".join(formatted_list)
    prompt = f"{joined_sentence}\n{new_prompt}"
    return prompt
def split_sentence(sentence): # splits sentence, might have to change for better performance
    if "and" in sentence:
        return sentence.split("and")
    elif "then" in sentence:
        return sentence.split("then")
    else:
        return [sentence]
### FINAL FUNCTION
def function_llm_pipeline(prompt, system):
  prompts = []
  responses = []
  sentence = prompt
  splitted_sentence = split_sentence(sentence)
  for i, sentence in enumerate(splitted_sentence):
      if i == 0: 
        prompt = format_prompt(sentence, None, None, i)
        #print(prompt)
        response = function_llm(f"Correctly solve this task: {prompt}", system)
        print(response)
        prompts.append(f"{sentence}")
        responses.append(response)
        #print(sentence)
      else:
        prompt = format_prompt(sentence, prompts, responses, i)
        #print(prompt)
        response = function_llm(f"Correctly solve this task: {prompt}", system)
        print(response)
        prompts.append(f"{sentence}")
        responses.append(response)
        #print(sentence)
  return responses
