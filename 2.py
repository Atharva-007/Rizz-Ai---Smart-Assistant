import time
from vision import capture_frame_from_webcam
from ai_assistant import listen_to_speech, speak, get_weather, get_mistral_response

def main():
    """Main function to integrate both speech recognition and video processing."""
    # Start the webcam video stream and object detection
    detected_objects_gen = capture_frame_from_webcam()

    # Start listening for speech and responding
    while True:
        query = listen_to_speech()
        if query:
            response = ""

            # Handle speech queries
            if "weather" in query.lower():
                city = query.split("weather in")[-1].strip()
                response = get_weather(city)
           
            elif "what do you see" in query.lower():
                # Get the latest detected objects from the webcam feed
                detected_objects = next(detected_objects_gen)
                response = f"I see the following objects: {detected_objects}"
            else:
                response = get_mistral_response(query)

            print(f"AI Response: {response}")
            speak(response)  # Speak the response
        
        time.sleep(1)

if __name__ == "__main__":
    main()
