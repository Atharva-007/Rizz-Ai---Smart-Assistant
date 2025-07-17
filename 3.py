import cv2
import speech_recognition as sr
import pyttsx3
import subprocess
import threading
import time
import numpy as np
import logging
from ultralytics import YOLO

# ==== CONFIG ====
model = YOLO("yolov8n.pt")
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

recognizer = sr.Recognizer()
tts = pyttsx3.init()
is_running = True
last_frame = None

# ==== TTS ====
def speak(text):
    print(f"\n🧠 AI: {text}\n")
    tts.say(text)
    tts.runAndWait()

# ==== TEXT AI (Mistral) ====
def generate_text_response(prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", "mistral"],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30
        )
        return result.stdout.decode().strip()
    except:
        return "Something went wrong with the AI response."

# ==== VISION AI (LLaVA) ====
def generate_vision_response(image, prompt):
    try:
        path = "frame.jpg"
        cv2.imwrite(path, image)
        result = subprocess.run(
            ["ollama", "run", "llava", "-i", path],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30
        )
        return result.stdout.decode().strip()
    except:
        return "I couldn't analyze the visual feed."

# ==== PROMPT FILTER ====
def is_vision_prompt(text):
    text = text.lower()
    return any(x in text for x in [
        "what do you see", "how do i look", "what am i wearing",
        "can you see me", "describe the scene", "analyze my appearance"
    ])
    
def is_relevant_object_present(results, query):
    query = query.lower()
    if "look" in query or "wearing" in query or "shirt" in query:
        return any(obj.cls == 0 for obj in results[0].boxes)  # 'person' class
    return False

# ==== LISTEN + RESPOND ====
# ==== LISTEN + RESPOND ====
def listen_and_respond():
    global last_frame
    with sr.Microphone() as source:
        while is_running:
            print("\n🎤 Listening...")
            try:
                audio = recognizer.listen(source, timeout=5)
                query = recognizer.recognize_google(audio)
                print(f"\n🗣️ You said: {query}")

                if is_vision_prompt(query):  # Check if the query is related to vision
                    if last_frame is not None:
                        # Send to LLaVA for vision-based response (e.g., t-shirt color, appearance)
                        response = generate_vision_response(last_frame, query)
                    else:
                        response = "My camera isn’t ready yet."
                else:
                    response = generate_text_response(query)

                speak(response)

            except sr.WaitTimeoutError:
                print("⚠️ Listening timed out...")
            except sr.UnknownValueError:
                print("🤖 Didn't catch that.")
            except Exception as e:
                print("❌ Error:", e)

# ==== VISION LOOP ====
def vision_loop():
    global last_frame
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ Cannot access webcam.")
        return

    while is_running:
        ret, frame = cap.read()
        if not ret:
            continue

        last_frame = frame.copy()
        results = model(frame)
        annotated = results[0].plot()

        cv2.imshow("👁️ JARVIS Vision", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ==== MAIN ====
if __name__ == "__main__":
    threading.Thread(target=vision_loop, daemon=True).start()
    listen_and_respond()
