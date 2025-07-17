import subprocess
import cv2

def capture_image(filename="frame.jpg", camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(filename, frame)
        print("📸 Frame captured for LLaVA.")
    else:
        print("❌ Failed to capture frame.")
    cap.release()

def query_llava(prompt, image_path="frame.jpg"):
    result = subprocess.run(
        ["ollama", "run", "llava"],
        input=f"<image:{image_path}>\n{prompt}\n",
        capture_output=True,
        text=True
    )
    print("🧪 STDOUT:\n", result.stdout.strip())
    print("⚠️ STDERR:\n", result.stderr.strip())
    return result.stdout.strip()

if __name__ == "__main__":
    capture_image()
    response = query_llava("What do you see in this image?")
    print("🤖 LLaVA says:\n", response)
