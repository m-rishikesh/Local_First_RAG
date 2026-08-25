from google import genai
import requests,os
import time
client = genai.Client()

while True:
    prompt = input(f"\nYou: ")
    if prompt.lower() in ["exit", "quit"]:
        break
    rag = requests.post("http://localhost:8000/search",json={
        "query":f"{prompt}",
        "top_k":2
    }
    )
    response_rag = rag.json()
    response_text = ""
    for result in response_rag["results"]:
        response_text += result["text"] + "\n\n"
    response_text += "Return Response from the above details only and if no details are there please say no details are provided for your query. \n\n"
    response_text += f"query: {prompt}"

    api_key = os.environ["OPENROUTER_API_KEY"]
    for attempt in range(3):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-oss-120b",
                    "messages": [
                        {
                            "role": "user",
                            "content": response_text
                        }
                    ]
                }
            )
            data = response.json()
            if response.status_code == 200:
                data = response.json()
                answer = data["choices"][0]["message"]["content"]
                print(answer)
            else:
                print("Request failed")
                print("HTTP status:", response.status_code)
                print("Error:", response.text)
                break
            break
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                wait = 2 ** attempt
                print(f"Gemini busy. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
