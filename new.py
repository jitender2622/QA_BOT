import google.generativeai as genai
import os

# 1. Paste your API Key here
api_key = "AIzaSyBhHXTdbe8E40zJ5g_aFqt3QJvTPDkqOAk"

# 2. Configure the library
genai.configure(api_key=api_key)

print("--- Available Text Generation Models ---")
try:
    # 3. List all models
    for m in genai.list_models():
        # Only show models that can generate text (Chat)
        if 'generateContent' in m.supported_generation_methods:
            print(f"Model Name: {m.name}")
            
except Exception as e:
    print(f"Error: {e}")