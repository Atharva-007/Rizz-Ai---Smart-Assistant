import argparse
import sys
from rizz_ai import __version__
from rizz_ai.services.vision import VisionService
from rizz_ai.services.stt import DummySTT
from rizz_ai.services.tts import PyTTS

def main():
    parser = argparse.ArgumentParser(prog="rizz-ai")
    parser.add_argument("action", nargs="?", default="help", choices=["vision-test", "version", "help"], help="Action to run")
    args = parser.parse_args()

    if args.action == "version":
        print(f"rizz-ai version {__version__}")
        sys.exit(0)

    if args.action == "vision-test":
        vs = VisionService()
        vs.capture_once()
        sys.exit(0)

    parser.print_help()


if __name__ == "__main__":
    main()