import requests
import json

ollama_url = "https://llm.cialabs.org/api/generate"

def summarize_text(text):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3.1",
        "prompt": f"summarize the following text: {text}"
    }
    
    try:
        response = requests.post(ollama_url, headers=headers, json=data, stream=True)
        summary = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    result = json.loads(line)
                    if result.get("done"):
                        break
                    summary += result.get("response", "")
                except ValueError as e:
                    print(f"Error during parsing: {str(e)}")
                    return None

        return summary.strip() if summary else "No summary generated"
    except Exception as e:
        print(f"Error during summarization: {str(e)}")
        return None
