import cv2
import numpy as np
from transformers import pipeline
import matplotlib.pyplot as plt
import threading

# Load the conversational AI model
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Press 'q' to quit.")

# Initialize conversation history
conversation_history = []
ai_response = ""

def get_user_input():
    global ai_response
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'q':
            break
        conversation_history.append(user_input)
        response = chatbot(" ".join(conversation_history), max_length=100, num_return_sequences=1)
        ai_response = response[0]['generated_text']
        conversation_history.append(ai_response)

# Start the user input thread
input_thread = threading.Thread(target=get_user_input)
input_thread.start()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Convert frame to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the resulting frame using matplotlib
    plt.imshow(frame_rgb)
    plt.axis('off')  # Hide axes
    plt.pause(0.001)  # Pause to allow the plot to update

    # Overlay AI response on the frame
    plt.text(10, 10, ai_response, color='white', fontsize=12, backgroundcolor='black')

    # Check for exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
plt.close()