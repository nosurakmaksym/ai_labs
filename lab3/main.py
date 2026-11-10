from google import genai
import os

API_KEY = "AQ.Ab8RN6Ie3KTF_7_8LiUv-j5PygM_-k6SV5X_lOMpIfbASEhudg"

client = genai.Client(api_key=API_KEY)
MODEL = "gemini-3.5-flash-lite"          

while True:
    prompt = input("\n>> ").strip()

    if prompt.lower() in "q":
        print("До побачення!")
        break

    if not prompt:
        continue

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )
        print("\nGemini:")
        print(response.text)
    except Exception as e:
        print(f"\nПомилка: {e}")