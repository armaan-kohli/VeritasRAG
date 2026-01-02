import google.generativeai as genai

# Configure your API key
genai.configure(api_key="AIzaSyBBOftGc6MK8fcEq69WRqJCkRoLyuSsWDY")

print("Available models for your API key:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"Model: {m.name} | Methods: {m.supported_generation_methods}")