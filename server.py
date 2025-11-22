import os
import json
import requests
import time
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# --- Configuration ---
# NOTE: The API key below has been updated with the key you provided.
# Using environment variables (GEMINI_API_KEY) is still the recommended best practice for security,
# but using this value as the default ensures the app runs immediately.
PLACEHOLDER_KEY = "AIzaSyBiKJJPloihS1aFZABMu8Cvy2mgiK8oGWU"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", PLACEHOLDER_KEY)

if not GEMINI_API_KEY or GEMINI_API_KEY == PLACEHOLDER_KEY:
    # Use a print statement instead of raising an error for clearer console output
    print("------------------------------------------------------------------")
    print("WARNING: Using hardcoded API key or GEMINI_API_KEY environment variable is not set.")
    print("For security, it is highly recommended to set the API key using an environment variable.")
    print("------------------------------------------------------------------")
else:
    print("\n------------------------------------------------------------------")
    print("  Gemini API Key Status: Loaded from Environment Variable")
    print("------------------------------------------------------------------")

app = Flask(__name__)
# Enable CORS for local development, allowing the client (even if opened directly) to talk to the server
CORS(app) 

# Base URLs for the Gemini API
GEMINI_SUMMARIZE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={GEMINI_API_KEY}"
GEMINI_TTS_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key={GEMINI_API_KEY}"

# --- Helper Functions ---
def fetch_gemini_api(url, payload):
    """Handles the POST request to the Gemini API with exponential backoff."""
    global GEMINI_API_KEY, PLACEHOLDER_KEY
    # The check below is now less critical since you provided a real key,
    # but it remains to alert if the env var is not set and the default key is used.
    # We remove the EnvironmentError raise to allow the updated PLACEHOLDER_KEY to work.

    headers = {
        'Content-Type': 'application/json'
    }
    max_retries = 5
    delay = 1.0  # Initial delay in seconds

    for i in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.HTTPError as e:
            # Check for Rate Limit (429)
            if response.status_code == 429 and i < max_retries - 1:
                print(f"Rate limit exceeded (429). Retrying in {delay:.2f}s...")
                time.sleep(delay)
                delay *= 2
                continue
            # Handle other HTTP errors
            error_details = response.json() if response.text else "No error message provided."
            print(f"HTTP Error {response.status_code}: {error_details}")
            # If the error is 400 or 403, it's highly likely an invalid API key
            if response.status_code in [400, 403]:
                error_message = f"API Key likely invalid or not configured correctly. Status {response.status_code}. Details: {error_details}"
                raise requests.exceptions.HTTPError(error_message, response=response)
            
            raise requests.exceptions.HTTPError(f"API call failed with status {response.status_code}", response=response)
        except requests.exceptions.RequestException as e:
            # Handle general request errors (e.g., timeout, network failure)
            if i < max_retries - 1:
                print(f"Request failed ({e}). Retrying in {delay:.2f}s...")
                time.sleep(delay)
                delay *= 2
                continue
            raise e

# --- Routes ---

@app.route('/summarize', methods=['POST'])
def summarize():
    """Endpoint to handle text summarization via Gemini 2.5 Flash."""
    try:
        data = request.get_json()
        summary_prompt = data.get('summaryPrompt')

        if not summary_prompt:
            return jsonify({"error": "Missing summaryPrompt in request body"}), 400

        # System prompt is defined on the server side for consistency
        system_prompt = "You are an expert academic revision assistant. Your goal is to provide concise, accurate, and easily digestible study notes. Always output a clean list of bullet points and identify keywords within the summary using a <mark> tag."

        payload = {
            "contents": [{"parts": [{"text": summary_prompt}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
        }

        api_response = fetch_gemini_api(GEMINI_SUMMARIZE_URL, payload)
        
        # Extract the generated text
        text_result = api_response.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')

        if text_result:
            return jsonify({"summary": text_result})
        else:
            print(f"API Response Error: {api_response}")
            return jsonify({"error": "Failed to generate summary from the model."}), 500

    except requests.exceptions.HTTPError as e:
        # Pass the HTTP error message back to the client
        return jsonify({"error": f"API Request Failed: {e}"}), 500
    except Exception as e:
        print(f"Summarization Error: {e}")
        return jsonify({"error": f"Server Error during summarization: {e}"}), 500

@app.route('/read-aloud', methods=['POST'])
def read_aloud():
    """Endpoint to handle Text-to-Speech via Gemini TTS."""
    try:
        data = request.get_json()
        text_to_speak = data.get('textToSpeak')
        
        if not text_to_speak:
            return jsonify({"error": "Missing textToSpeak in request body"}), 400

        payload = {
            "contents": [{"parts": [{"text": f"Say in a clear, informative voice: {text_to_speak}"}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": "Kore"}
                    }
                }
            },
        }

        api_response = fetch_gemini_api(GEMINI_TTS_URL, payload)

        # Extract the audio data (base64 encoded PCM16) and mimeType
        part = api_response.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0]
        audio_data = part.get('inlineData', {}).get('data')
        mime_type = part.get('inlineData', {}).get('mimeType') # Should be audio/L16;rate=24000

        if audio_data and mime_type and mime_type.startswith("audio/L16"):
            # Return the base64 audio data and mime type for the client to process
            return jsonify({"audioData": audio_data, "mimeType": mime_type})
        else:
            print(f"TTS API Response Error: {api_response}")
            return jsonify({"error": "Invalid audio data received from API or model failed to generate audio."}), 500

    except requests.exceptions.HTTPError as e:
        # Pass the HTTP error message back to the client
        return jsonify({"error": f"API Request Failed: {e}"}), 500
    except Exception as e:
        print(f"TTS Error: {e}")
        return jsonify({"error": f"Server Error during TTS generation: {e}"}), 500

@app.route('/')
def serve_index():
    """Serve the index.html file."""
    # Assuming index.html is in the same directory as server.py
    return send_file('index.html')


if __name__ == '__main__':
    # Add time import for exponential backoff helper
    import time
    
    # We moved the key check and status message outside the main block for clearer output
    print("\n------------------------------------------------------------------")
    print("  Starting Flask Server...")
    print("  Access the app at: http://127.0.0.1:5000/")
    print("  Stop the server with CTRL+C")
    print("------------------------------------------------------------------\n")
    # If debug=True, Flask automatically imports time again. 
    # Since it's already imported at the top, this is safe.
    app.run(debug=True, host='0.0.0.0', port=5000)
