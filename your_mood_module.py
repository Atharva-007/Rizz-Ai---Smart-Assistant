# your_mood_module.py
import random

def detect_tone_and_generate_response(user_input, detected_object):
    # Simple tone detection (you can enhance this with NLP models)
    tone = "neutral"

    rude_keywords = ["stupid", "idiot", "useless", "dumb"]
    funny_keywords = ["joke", "lol", "funny", "meme"]
    happy_keywords = ["great", "awesome", "good", "cool"]
    sad_keywords = ["sad", "depressed", "tired", "lonely"]
    excited_keywords = ["yay", "omg", "wow"]

    tone_categories = {
        "rude": rude_keywords,
        "funny": funny_keywords,
        "happy": happy_keywords,
        "sad": sad_keywords,
        "excited": excited_keywords,
        # Add more categories here if needed
    }

    for category, keywords in tone_categories.items():
        if any(word in user_input.lower() for word in keywords):
            tone = category
            break

    # Response dictionary for different tones
    responses = {
        "rude": [
            f"Easy there, champ. I’m just a helpful bot, not your punching bag.",
            f"Careful, your attitude might overheat my sarcasm detector."
        ],
        "funny": [
            f"Haha, you're clearly the class clown today! I like it.",
            f"You're on a roll, keep those jokes coming!"
        ],
        "happy": [
            f"Love that energy! Let’s keep it going.",
            f"You're in a great mood! So, I see {detected_object} right now."
        ],
        "sad": [
            f"Hey, whatever you're going through, I'm here for you.",
            f"It's okay to feel down sometimes. Wanna talk about {detected_object}?"
        ],
        "excited": [
            f"Whoa! That energy is electric. Let's channel it!",
            f"You're fired up! Let's take a look at this {detected_object}."
        ],
        "neutral": [
            f"Right now, I’m seeing {detected_object}. What do you want to do with it?",
            f"Hmm, looks like {detected_object}. Need help with it?"
        ]
    }

    return random.choice(responses.get(tone, responses["neutral"]))
