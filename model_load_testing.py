# Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("text-generation", model="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF")
# print(pipe("Hello, how are you?", max_new_tokens=50))

import lmstudio as lms

model = lms.llm()
print(model.respond("What is the meaning of life?"))
