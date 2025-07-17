import speech_recognition as sr
import pyttsx3
import subprocess
import time
import requests
import json
import os
import cv2
from datetime import datetime
from ultralytics import YOLO

# === CONFIG ===
AI_NAME = "rohit"
SUMMARY_FILE = "summary.json"
model = YOLO("yolov8n.pt")
engine = pyttsx3.init()


# === VOICE FUNCTIONS ===
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
            command = recognizer.recognize_google(audio)
            print(f"🗣️ You said: {command}")
            return command
        except sr.UnknownValueError:
            print("⚠️ Could not understand.")
            return None
        except sr.RequestError:
            print("⚠️ Network error.")
            return None


# === SUMMARY FUNCTIONS ===
def generate_detailed_summary(objects, people_count):
    summary = f"Detected {people_count} person(s) in the scene.\n"
    for i in range(people_count):
        summary += f"Person {i+1} is wearing:\n"
    summary += "\nDetected objects and their details:\n"
    for obj in objects:
        summary += f"The detected object is a {obj['name']}.\n"
    return summary

def generate_paragraph_summary(objects, people_count):
    summary = f"There are {people_count} person(s) in the scene. "
    for i in range(people_count):
        summary += f"Person {i+1} appears to be wearing normal clothes. "
    summary += "Some objects detected include: "
    for obj in objects:
        summary += f"a {obj['name']}, "
    return summary.strip().rstrip(',') + ". This is a brief summary of the scene."


# === VISION FUNCTION ===
def capture_frame_from_webcam():
    cap = cv2.VideoCapture(0)
    frame_count = 0
    print("📷 Starting webcam...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Couldn't read from webcam.")
            break

        results = model(frame)
        detections = results[0].boxes

        objects_detected = []
        people_count = 0

        for box in detections:
            class_name = box.cls
            confidence = box.conf
            class_names = model.names
            detected_class_name = class_names[int(class_name)]

            if confidence > 0.5:
                if detected_class_name == 'person':
                    people_count += 1
                objects_detected.append({'name': detected_class_name, 'confidence': float(confidence)})

        detailed_summary = generate_detailed_summary(objects_detected, people_count)
        paragraph_summary = generate_paragraph_summary(objects_detected, people_count)

        summary_data = {
            "timestamp": str(datetime.now()),
            "objects": [obj['name'] for obj in objects_detected],
            "detailed_summary": detailed_summary,
            "paragraph_summary": paragraph_summary
        }

        with open(SUMMARY_FILE, "w") as json_file:
            json.dump(summary_data, json_file, indent=4)

        print("✅ Summary saved.")
        print(paragraph_summary)

        frame_count += 1
        if frame_count >= 5:
            break

    cap.release()
    print("🛑 Webcam stopped.")


# === OTHER UTILITIES ===
def read_summary():
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, "r") as file:
            return json.load(file)
    return None

def delete_summary():
    if os.path.exists(SUMMARY_FILE):
        os.remove(SUMMARY_FILE)

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
            print(f"❌ Mistral error: {error}")
        return response.strip() if response else "No response from model."
    except Exception as e:
        return f"Error: {e}"


# === AI BOT LOOP ===
def ai_bot():
    ai_active = False
    while True:
        query = listen_to_speech()
        if query:
            query_lower = query.lower()
            if AI_NAME.lower() in query_lower and not ai_active:
                ai_active = True
                speak(f"Hello, I am {AI_NAME}. How can I assist you today?")
                print("✅ AI activated.")
                capture_frame_from_webcam()

            elif "bye" in query_lower or "meet you later" in query_lower:
                speak("Goodbye, see you later!")
                print("👋 AI deactivated.")
                delete_summary()
                ai_active = False
                continue

            if ai_active:
                summary = read_summary()
                if summary and "vision" in query_lower:
                    response = f"Here's what I see: {summary.get('paragraph_summary', 'No details')}"
                else:
                    response = get_mistral_response(query)

                print(f"🤖 {response}")
                speak(response)

                if summary and "vision" not in query_lower:
                    delete_summary()

        time.sleep(1)

if __name__ == "__main__":
    ai_bot()
