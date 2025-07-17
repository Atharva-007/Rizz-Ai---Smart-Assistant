import speech_recognition as sr
import pyttsx3
import subprocess
import time
import requests
import json
import os

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Global config
AI_NAME = "rizz"
SUMMARY_FILE = "summary.json"
vision_process = None  # Background vision process

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen_to_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            print("🗣️ You said:")
            command = recognizer.recognize_google(audio)
            print(f"  {command}")
            return command
        except sr.UnknownValueError:
            print("⚠️ Sorry, I could not understand your speech.")
            return None
        except sr.RequestError:
            print("⚠️ Could not request results, check your internet connection.")
            return None

def get_crypto_price(crypto):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto}&vs_currencies=usd"
    response = requests.get(url)
    data = response.json()
    return f"The current price of {crypto} is ${data[crypto]['usd']}."

def get_weather(city):
    API_KEY = "a4e655b2cd3c3242e535965f01e2d0dd"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    if data["cod"] == 200:
        temp = data["main"]["temp"]
        description = data["weather"][0]["description"]
        return f"The temperature in {city} is {temp}°C with {description}."
    else:
        return "Sorry, I couldn't get the weather data."

def get_mistral_response(query):
    try:
        result = subprocess.run(
            ["ollama", "run", "mistral", query],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        response = result.stdout.decode()
        error = result.stderr.decode()

        if error:
            print(f"❌ Error with Mistral model: {error}")
        if response:
            return response.strip()
        else:
            return "Sorry, I couldn't get a response from the model."
    except Exception as e:
        print(f"❌ Error: {e}")
        return "I encountered an error while processing the request."

def read_summary_from_json():
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, "r") as file:
            return json.load(file)
    return None

def delete_summary_from_json():
    if os.path.exists(SUMMARY_FILE):
        os.remove(SUMMARY_FILE)

def start_recording():
    global vision_process
    print("🎥 Starting vision recording...")
    vision_process = subprocess.Popen(["python", "vision.py"])

def stop_recording():
    global vision_process
    print("🛑 Stopping vision recording...")
    if vision_process:
        vision_process.terminate()
        vision_process = None
    delete_summary_from_json()

def ai_bot():
    ai_active = False
    
    while True:
        query = listen_to_speech()
        if query:
            query_lower = query.lower()
            
            if AI_NAME.lower() in query_lower and not ai_active:
                ai_active = True
                speak(f"Hello, I am {AI_NAME}. How can I assist you today?")
                print("AI activated.")
                start_recording()
            
            elif "bye" in query_lower or "meet you later" in query_lower:
                ai_active = False
                speak("Goodbye, see you later!")
                print("AI deactivated.")
                stop_recording()
            
            if ai_active:
                summary = read_summary_from_json()
                if summary:
                    if "vision" in query_lower:
                        response = f"Here's what I see: {summary.get('paragraph_summary', 'No details available')}."
                    else:
                        response = get_mistral_response(query)
                else:
                    response = get_mistral_response(query)
                
                print(f"🧠 AI: {response}")
                speak(response)

                if summary and "vision" not in query_lower:
                    delete_summary_from_json()

        time.sleep(1)

if __name__ == "__main__":
    ai_bot()
