import cv2
import json
from datetime import datetime
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

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
    summary = summary.strip().rstrip(',') + ". This is a brief summary of the scene."
    return summary

def capture_frame_from_webcam():
    cap = cv2.VideoCapture(1)
    frame_count = 0

    print("📷 Starting webcam...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Couldn't read from webcam.")
            break

        print("📸 Frame captured.")

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

        with open("summary.json", "w") as json_file:
            json.dump(summary_data, json_file, indent=4)

        print("✅ Summary saved.")
        print(paragraph_summary)

        frame_count += 1
        if frame_count >= 5:
            break

    cap.release()
    print("🛑 Webcam stopped.")

if __name__ == "__main__":
    capture_frame_from_webcam()
