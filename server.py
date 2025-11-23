import os
import json
import requests
import time
from flask import Flask, request, jsonify
# We need to import send_file, but removed the root route that needed it for live hosting
from flask_cors import CORS 

# --- Configuration ---
# BEST PRACTICE: API Key should be set via environment variable for security.
# The PLACEHOLDER_KEY is only a fail-safe for local testing.
PLACEHOLDER_KEY = "AIzaSyBiKJJPloihS1aFZABMu8Cvy2mgiK8oGWU"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", PLACEHOLDER_KEY)

# --- Flask Initialization and CORS Configuration ---
app = Flask(__name__)

# Live Hosting Configuration:
# 1. In a live environment where the frontend is hosted separately (e.g., Netlify/Vercel), 
#    we must allow Cross-Origin Resource Sharing (CORS).
# 2. For simplicity and broad compatibility with separate static hosting, we allow all origins ('*').
#    If you knew the exact final domain (e.g., 'https://your-notes-app.com'), 
#    you would set CORS(app, origins=['https://your-notes-app.com']).
CORS(app, resources={r"/*": {"origins": "*"}}) 

# Base URLs for the Gemini API
GEMINI_SUMMARIZE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={GEMINI_API_KEY}"
GEMINI_TTS_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key={GEMINI_API_KEY}"

# --- Helper Functions ---
def fetch_gemini_api(url, payload):
    """Handles the POST request to the Gemini API with exponential backoff."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == PLACEHOLDER_KEY:
         raise EnvironmentError("GEMINI_API_KEY is not set. Please configure it in your hosting environment.")
         
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
            # If the error is 400 or 403, it's highly likely an invalid API key
            if response.status_code in [400, 403]:
                error_message = f"API Key invalid or rate limit exceeded. Status {response.status_code}. Details: {error_details}"
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

    except EnvironmentError as e:
        return jsonify({"error": str(e)}), 500
    except requests.exceptions.HTTPError as e:
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

    except EnvironmentError as e:
        return jsonify({"error": str(e)}), 500
    except requests.exceptions.HTTPError as e:
        return jsonify({"error": f"API Request Failed: {e}"}), 500
    except Exception as e:
        print(f"TTS Error: {e}")
        return jsonify({"error": f"Server Error during TTS generation: {e}"}), 500

# Removed the @app.route('/') to serve index.html. 
# In a standard web host setup, the frontend (index.html) is served by a static host (like Netlify/Vercel), 
# and the Python backend only serves the API routes.

if __name__ == '__main__':
    # Use environment variable for the port, defaulting to 5000 for local development.
    # Hosting platforms like Render/Heroku will set the PORT variable automatically.
    port = int(os.environ.get('PORT', 5000))
    
    # We use host='0.0.0.0' to listen on all public IPs, which is required for hosting providers.
    app.run(host='0.0.0.0', port=port, debug=False) # debug=False for production safety
