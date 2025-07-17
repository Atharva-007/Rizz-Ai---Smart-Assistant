import requests
import json
import os
import re
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from time import time
from textblob import TextBlob
import calendar
from collections import defaultdict
import numpy as np
from io import BytesIO
import base64
import csv
from uuid import uuid4
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import glob
import random

class NovaCompanion:
    def __init__(self):
        self.memory_file = "nova_memory.json"
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = "gemma3:4b"
        self.max_conversations = 100
        self.request_delay = 0.5
        self.last_request_time = 0
        self.export_dir = "nova_exports"
        self.model_dir = "nova_models"
        
        os.makedirs(self.export_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.reminder_handlers = {
            "every": self.handle_recurring_reminder,
            "daily": self.handle_daily_reminder,
            "weekly": self.handle_weekly_reminder,
            "monthly": self.handle_monthly_reminder
        }
        
        self.load_memory()
        self.mood_predictor = None
        self.load_mood_model()
        
        # Initialize new session
        self.memory["sessions"] = self.memory.get("sessions", 0) + 1
        self.memory["last_session"] = datetime.now().isoformat()
        
        # Check for daily mood reset
        self.check_daily_reset()
        
        # Check for reminders
        self.check_reminders()

    def _get_default_memory(self):
        return {
            "conversations": [],
            "user_info": {},
            "topics": {},
            "stats": {
                "total_messages": 0,
                "sessions": 0
            },
            "daily_mood": {
                "positive": 0,
                "neutral": 0,
                "negative": 0,
                "mood_changes": []
            },
            "historical_mood": {},
            "journal": [],
            "reminders": []
        }

    def load_memory(self):
        """Safe memory loading with error handling"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, "r") as f:
                    self.memory = json.load(f)
                # Prune old conversations if needed
                if len(self.memory.get("conversations", [])) > self.max_conversations:
                    self.memory["conversations"] = self.memory["conversations"][-self.max_conversations:]
            else:
                self.memory = self._get_default_memory()
        except Exception as e:
            print(f"⚠️ Memory load error: {e}")
            self.memory = self._get_default_memory()

    def save_memory(self):
        """Atomic memory save with error handling"""
        try:
            temp_file = f"{self.memory_file}.tmp"
            with open(temp_file, "w") as f:
                json.dump(self.memory, f, indent=2)
            os.replace(temp_file, self.memory_file)
        except Exception as e:
            print(f"⚠️ Failed to save memory: {e}")

    def _rate_limit(self):
        """Enforce minimal delay between requests"""
        elapsed = time() - self.last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request_time = time()

    def analyze_sentiment(self, text):
        """Basic sentiment analysis"""
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0.3:
            return "positive"
        elif analysis.sentiment.polarity < -0.3:
            return "negative"
        return "neutral"

    def detect_topics(self, text):
        """Simple topic detection"""
        topics = []
        text_lower = text.lower()
        
        topic_keywords = {
            "technology": ["computer", "phone", "ai", "tech", "program"],
            "food": ["eat", "food", "dinner", "lunch", "restaurant"],
            "weather": ["weather", "rain", "sunny", "temperature"],
            "entertainment": ["movie", "music", "game", "book"],
            "work": ["work", "job", "meeting", "boss"],
            "family": ["mom", "dad", "parents", "sibling", "child"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics if topics else ["general"]

    def check_daily_reset(self):
        """Reset daily mood tracker if new day"""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.memory.get("current_date") != today:
            self.memory["current_date"] = today
            self.memory["daily_mood"] = {
                "positive": 0,
                "neutral": 0,
                "negative": 0,
                "mood_changes": []
            }
            self.save_memory()

    def track_mood(self, sentiment):
        """Update mood statistics"""
        self.check_daily_reset()
        
        # Update daily counts
        self.memory["daily_mood"][sentiment] += 1
        
        # Record mood change with timestamp
        self.memory["daily_mood"]["mood_changes"].append({
            "time": datetime.now().isoformat(),
            "mood": sentiment,
            "intensity": abs(TextBlob(self.memory["conversations"][-1]["user"]).sentiment.polarity)
        })
        
        # Update historical mood (weekly)
        week_num = datetime.now().isocalendar()[1]
        year = datetime.now().year
        week_key = f"{year}-W{week_num}"
        
        if "historical_mood" not in self.memory:
            self.memory["historical_mood"] = defaultdict(lambda: {
                "positive": 0,
                "neutral": 0,
                "negative": 0
            })
        
        self.memory["historical_mood"][week_key][sentiment] += 1

    def generate_mood_chart(self, time_frame='week'):
        """Generate mood visualization as base64 encoded image"""
        try:
            plt.figure(figsize=(8, 4))
            
            if time_frame == 'day':
                data = self.memory["daily_mood"]
                moods = ["positive", "neutral", "negative"]
                counts = [data[m] for m in moods]
                title = "Today's Mood Distribution"
                
                plt.pie(counts, labels=[f"{m} ({c})" for m,c in zip(moods, counts)],
                        colors=['#4CAF50', '#FFC107', '#F44336'],
                        autopct='%1.1f%%')
            else:
                if not self.memory.get("historical_mood"):
                    return None
                
                weeks = sorted(self.memory["historical_mood"].keys())[-4:]
                moods = ["positive", "neutral", "negative"]
                counts = np.array([[self.memory["historical_mood"][w][m] for w in weeks] for m in moods])
                title = "Weekly Mood Trend"
                
                for i, mood in enumerate(moods):
                    plt.plot(weeks, counts[i], marker='o', label=mood.capitalize())
                plt.legend()
                plt.xticks(rotation=45)

            plt.title(title)
            plt.tight_layout()
            
            # Save to buffer
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            
            # Encode as base64
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            return f"data:image/png;base64,{img_base64}"
        
        except Exception as e:
            print(f"Chart generation error: {e}")
            return None

    def get_mood_summary(self, detailed=False):
        """Enhanced mood report with visualization"""
        if not self.memory["daily_mood"]["mood_changes"]:
            return "No mood data available today."
        
        daily = self.memory["daily_mood"]
        total = sum(daily[s] for s in ["positive", "neutral", "negative"]) or 1
        
        summary = f"📊 Today's Mood Summary ({self.memory['current_date']}):\n"
        summary += f"- 😊 Positive: {daily['positive']} ({daily['positive']/total:.0%})\n"
        summary += f"- 😐 Neutral: {daily['neutral']} ({daily['neutral']/total:.0%})\n"
        summary += f"- 😞 Negative: {daily['negative']} ({daily['negative']/total:.0%})\n"
        
        # Add trend if enough data
        if len(daily["mood_changes"]) > 3:
            last_half = daily["mood_changes"][len(daily["mood_changes"])//2:]
            positive_trend = sum(1 for m in last_half if m["mood"] == "positive") / len(last_half)
            if positive_trend > 0.6:
                summary += "📈 Trend: Your mood has been improving!\n"
            elif positive_trend < 0.4:
                summary += "📉 Trend: Your mood has been declining.\n"
        
        if detailed:
            chart = self.generate_mood_chart('day')
            if chart:
                summary += f"\n![Mood Chart]({chart})"
            
            weekly_chart = self.generate_mood_chart('week')
            if weekly_chart:
                summary += f"\n\n![Weekly Mood Trend]({weekly_chart})"
        
        # Mood-topic correlation
        if self.memory["conversations"]:
            mood_topics = defaultdict(lambda: defaultdict(int))
            for conv in self.memory["conversations"]:
                if "sentiment" in conv and "topics" in conv:
                    for topic in conv["topics"]:
                        mood_topics[conv["sentiment"]][topic] += 1
            
            if mood_topics:
                summary += "\n\n🔍 Mood-Topic Correlations:"
                for mood in ["positive", "neutral", "negative"]:
                    if mood_topics[mood]:
                        top_topic = max(mood_topics[mood].items(), key=lambda x: x[1])
                        summary += f"\n- When discussing {top_topic[0]}, you tend to be {mood}"

        return summary

    def summarize_context(self):
        """Enhanced context summary with sentiment and topics"""
        if not self.memory["conversations"]:
            return "No prior conversation history."
        
        summary = "## Conversation Context ##\n"
        
        # User info
        if self.memory["user_info"]:
            summary += "\n### Known User Details ###\n"
            for key, value in self.memory["user_info"].items():
                summary += f"- {key}: {value}\n"
        
        # Recent conversations (last 5)
        summary += "\n### Recent Messages ###\n"
        for conv in self.memory["conversations"][-5:]:
            sentiment = conv.get("sentiment", "unknown")
            summary += f"[{conv['time'][11:19]} {sentiment.upper()}] You: {conv['user'][:60]}...\n"
        
        # Frequent topics
        if self.memory["topics"]:
            summary += "\n### Frequent Topics ###\n"
            for topic, count in sorted(self.memory["topics"].items(), key=lambda x: x[1], reverse=True)[:3]:
                summary += f"- {topic.title()} ({count}x mentioned)\n"
        
        return summary

    def generate_response(self, user_input):
        """Enhanced response generation with error handling"""
        self._rate_limit()
        
        try:
            # Analyze current input
            sentiment = self.analyze_sentiment(user_input)
            topics = self.detect_topics(user_input)
            
            # Update topic frequencies
            for topic in topics:
                self.memory["topics"][topic] = self.memory["topics"].get(topic, 0) + 1
            
            context = self.summarize_context()
            
            prompt = f"""
            [System: Nova - Friendly AI Companion]
            - Current Time: {datetime.now().strftime("%Y-%m-%d %H:%M")}
            - User Sentiment: {sentiment}
            - Detected Topics: {', '.join(topics)}
            
            {context}
            
            [Current Chat - Respond appropriately for {sentiment} sentiment]
            User: {user_input}
            Nova: """
            
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_ctx": 2048
                    }
                },
                timeout=10
            )
            response.raise_for_status()
            
            return response.json()["response"].strip()
            
        except requests.exceptions.RequestException as e:
            print(f"⚠️ API Error: {e}")
            return "I'm having some technical difficulties. Please try again in a moment."
        except Exception as e:
            print(f"⚠️ Unexpected error: {e}")
            return "Something unexpected happened. Let's try that again."

    def parse_reminder(self, user_input):
        """Enhanced reminder parsing with recurrence support"""
        # Check for recurring patterns first
        recurring_patterns = [
            (r"(every day|daily) (?:at|remind me to) (.+?) (?:at|on) (.+)", "daily"),
            (r"(every week|weekly) (?:at|remind me to) (.+?) (?:at|on) (.+)", "weekly"),
            (r"(every month|monthly) (?:at|remind me to) (.+?) (?:at|on) (.+)", "monthly"),
            (r"remind me to (.+?) every (.+)", "every")
        ]
        
        for pattern, freq in recurring_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                if freq == "every":
                    task = match.group(1).strip()
                    time_str = f"every {match.group(2).strip()}"
                else:
                    task = match.group(2).strip() if len(match.groups()) > 2 else match.group(1).strip()
                    time_str = match.group(3).strip() if len(match.groups()) > 2 else match.group(2).strip()
                return task, time_str, freq
        
        # Fall back to normal reminder parsing
        normal_patterns = [
            r"remind me to (.+?) (?:at|on) (.+)",
            r"set a reminder for (.+?) (?:at|on) (.+)",
            r"tell me to (.+?) (?:at|on) (.+)"
        ]
        
        for pattern in normal_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                task = match.group(1).strip()
                time_str = match.group(2).strip()
                return task, time_str, None
        
        return None, None, None

    def parse_time(self, time_str):
        """Convert natural time to datetime"""
        time_str = time_str.lower()
        now = datetime.now()
        
        # Handle relative times
        if "in " in time_str:
            match = re.search(r"in (\d+) (\w+)", time_str)
            if match:
                quantity, unit = match.groups()
                try:
                    quantity = int(quantity)
                    unit = unit.rstrip('s')  # remove plural
                    
                    if unit in ["minute", "min"]:
                        return now + timedelta(minutes=quantity)
                    elif unit in ["hour", "hr"]:
                        return now + timedelta(hours=quantity)
                    elif unit == "day":
                        return now + timedelta(days=quantity)
                    elif unit == "week":
                        return now + timedelta(weeks=quantity)
                except:
                    pass
        
        # Handle absolute times
        try:
            if "tomorrow" in time_str:
                tomorrow = now + timedelta(days=1)
                return tomorrow.replace(hour=12, minute=0, second=0)  # Default to noon
            
            day_names = [day.lower() for day in calendar.day_name]
            if any(day in time_str for day in day_names):
                for i, day in enumerate(day_names):
                    if day in time_str:
                        days_ahead = (i - now.weekday()) % 7
                        if days_ahead == 0:  # If today, schedule for next week
                            days_ahead = 7
                        return (now + timedelta(days=days_ahead)).replace(
                            hour=12, minute=0, second=0)
            
            # Try to parse exact time (e.g., "3pm")
            time_match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", time_str)
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2) or 0)
                period = time_match.group(3)
                
                if period == "pm" and hour < 12:
                    hour += 12
                elif period == "am" and hour == 12:
                    hour = 0
                
                # If time is in the past, assume next day
                result = now.replace(hour=hour, minute=minute, second=0)
                if result < now:
                    result += timedelta(days=1)
                
                return result
        
        except Exception as e:
            print(f"Time parsing error: {e}")
        
        return None

    def handle_recurring_reminder(self, task, time_str, frequency):
        """Create a recurring reminder"""
        base_time = self.parse_time(time_str)
        if not base_time:
            return False, "Sorry, I couldn't understand that time."
        
        if "reminders" not in self.memory:
            self.memory["reminders"] = []
        
        self.memory["reminders"].append({
            "id": str(uuid4()),
            "task": task,
            "time": base_time.isoformat(),
            "created": datetime.now().isoformat(),
            "completed": False,
            "recurring": True,
            "frequency": frequency,
            "last_triggered": None
        })
        
        self.save_memory()
        return True, f"✅ Recurring reminder set for {task} {frequency} starting {base_time.strftime('%A, %B %d at %I:%M %p')}."

    def handle_daily_reminder(self, task, time_str):
        """Create daily recurring reminder"""
        return self.handle_recurring_reminder(task, time_str, "daily")

    def handle_weekly_reminder(self, task, time_str):
        """Create weekly recurring reminder"""
        return self.handle_recurring_reminder(task, time_str, "weekly")

    def handle_monthly_reminder(self, task, time_str):
        """Create monthly recurring reminder"""
        return self.handle_recurring_reminder(task, time_str, "monthly")

    def add_reminder(self, task, time_str, frequency=None):
        """Enhanced reminder creation with recurrence support"""
        if frequency and frequency in self.reminder_handlers:
            return self.reminder_handlers[frequency](task, time_str)
        
        # Original reminder handling for one-time reminders
        remind_time = self.parse_time(time_str)
        if not remind_time:
            return False, "Sorry, I couldn't understand that time. Try something like '3pm' or 'in 2 hours'."
        
        if "reminders" not in self.memory:
            self.memory["reminders"] = []
        
        self.memory["reminders"].append({
            "id": str(uuid4()),
            "task": task,
            "time": remind_time.isoformat(),
            "created": datetime.now().isoformat(),
            "completed": False
        })
        
        self.save_memory()
        return True, f"✅ Reminder set for {task} at {remind_time.strftime('%A, %B %d at %I:%M %p')}."

    def check_reminders(self):
        """Enhanced reminder checking with recurrence support"""
        if "reminders" not in self.memory:
            return
        
        now = datetime.now()
        reminders_due = []
        
        for reminder in self.memory["reminders"]:
            if reminder.get("completed"):
                continue
                
            remind_time = datetime.fromisoformat(reminder["time"])
            
            # Handle recurring reminders
            if reminder.get("recurring"):
                last_triggered = (datetime.fromisoformat(reminder["last_triggered"]) 
                                if reminder.get("last_triggered") else None)
                
                # Check if it's time for the next occurrence
                if now >= remind_time and (not last_triggered or 
                    (reminder["frequency"] == "daily" and (now - last_triggered).days >= 1) or
                    (reminder["frequency"] == "weekly" and (now - last_triggered).days >= 7) or
                    (reminder["frequency"] == "monthly" and (now.year > last_triggered.year or 
                     now.month > last_triggered.month))):
                    
                    reminders_due.append(reminder)
                    reminder["last_triggered"] = now.isoformat()
                    continue
            
            # Handle one-time reminders
            elif remind_time <= now:
                reminders_due.append(reminder)
                reminder["completed"] = True
        
        if reminders_due:
            print("\n🔔 Reminders:")
            for reminder in reminders_due:
                freq = f" ({reminder['frequency']})" if reminder.get("recurring") else ""
                print(f"- {reminder['task']}{freq}")
            print()
        
        self.save_memory()

    def list_reminders(self, show_ids=False):
        """Enhanced reminder listing with optional IDs"""
        if "reminders" not in self.memory or not any(not r["completed"] for r in self.memory["reminders"]):
            return "No upcoming reminders."
        
        output = "📅 Upcoming Reminders:\n"
        now = datetime.now()
        
        for reminder in sorted(
            [r for r in self.memory["reminders"] if not r["completed"]], 
            key=lambda x: x["time"]
        ):
            remind_time = datetime.fromisoformat(reminder["time"])
            time_diff = remind_time - now
            
            # Time description
            if time_diff.days > 0:
                time_str = f"{time_diff.days} day{'s' if time_diff.days > 1 else ''} from now"
            elif time_diff.seconds > 3600:
                hours = time_diff.seconds // 3600
                time_str = f"{hours} hour{'s' if hours > 1 else ''} from now"
            elif time_diff.seconds > 60:
                minutes = time_diff.seconds // 60
                time_str = f"{minutes} minute{'s' if minutes > 1 else ''} from now"
            else:
                time_str = "now"
            
            # Recurrence info
            recur = f" [{reminder['frequency']}]" if reminder.get("recurring") else ""
            
            # ID display
            id_str = f" (ID: {reminder['id']})" if show_ids else ""
            
            output += f"- {reminder['task']}{recur}{id_str} ({time_str})\n"
        
        if show_ids:
            output += "\nUse these IDs with /modify or /delete commands"
        
        return output

    def modify_reminder(self, reminder_id, new_task=None, new_time=None):
        """Modify an existing reminder"""
        if "reminders" not in self.memory:
            return False, "No reminders exist"
        
        for reminder in self.memory["reminders"]:
            if reminder.get("id") == reminder_id:
                if new_task:
                    reminder["task"] = new_task
                if new_time:
                    new_time_parsed = self.parse_time(new_time)
                    if not new_time_parsed:
                        return False, "Invalid time format"
                    reminder["time"] = new_time_parsed.isoformat()
                
                self.save_memory()
                return True, "Reminder updated successfully"
        
        return False, "Reminder not found"

    def delete_reminder(self, reminder_id):
        """Delete a reminder"""
        if "reminders" not in self.memory:
            return False, "No reminders exist"
        
        initial_count = len(self.memory["reminders"])
        self.memory["reminders"] = [r for r in self.memory["reminders"] 
                                   if r.get("id") != reminder_id]
        
        if len(self.memory["reminders"]) < initial_count:
            self.save_memory()
            return True, "Reminder deleted"
        return False, "Reminder not found"

    def import_reminders(self, filepath):
        """Import reminders from CSV file"""
        try:
            with open(filepath, newline='') as f:
                reader = csv.DictReader(f)
                imported = 0
                
                for row in reader:
                    if not all(k in row for k in ['task', 'time']):
                        continue
                    
                    # Parse time or skip if invalid
                    remind_time = self.parse_time(row['time'])
                    if not remind_time:
                        continue
                    
                    # Create reminder
                    if "reminders" not in self.memory:
                        self.memory["reminders"] = []
                    
                    self.memory["reminders"].append({
                        "id": str(uuid4()),
                        "task": row['task'],
                        "time": remind_time.isoformat(),
                        "created": datetime.now().isoformat(),
                        "completed": False,
                        "recurring": row.get('recurring', '').lower() == 'true',
                        "frequency": row.get('frequency', ''),
                        "last_triggered": None
                    })
                    imported += 1
                
                self.save_memory()
                return True, f"Successfully imported {imported} reminders"
        
        except Exception as e:
            return False, f"Import failed: {str(e)}"

    def export_reminders(self):
        """Export reminders to CSV"""
        if "reminders" not in self.memory or not self.memory["reminders"]:
            return False, "No reminders to export"
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.export_dir}/reminders_export_{timestamp}.csv"
            
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'id', 'task', 'time', 'recurring', 'frequency'
                ])
                writer.writeheader()
                
                for reminder in self.memory["reminders"]:
                    writer.writerow({
                        'id': reminder['id'],
                        'task': reminder['task'],
                        'time': datetime.fromisoformat(reminder['time']).strftime('%Y-%m-%d %H:%M'),
                        'recurring': str(reminder.get('recurring', False)),
                        'frequency': reminder.get('frequency', '')
                    })
            
            return True, f"Reminders exported to {filename}"
        except Exception as e:
            return False, f"Export failed: {str(e)}"

    def add_journal_entry(self, entry_text, mood=None, tags=None):
        """Add a manual mood journal entry"""
        if "journal" not in self.memory:
            self.memory["journal"] = []
        
        entry = {
            "id": str(uuid4()),
            "timestamp": datetime.now().isoformat(),
            "entry": entry_text,
            "mood": mood or self.analyze_sentiment(entry_text),
            "tags": tags or []
        }
        
        self.memory["journal"].append(entry)
        self.save_memory()
        return entry

    def get_journal_entries(self, days=7):
        """Retrieve journal entries from specified period"""
        cutoff = datetime.now() - timedelta(days=days)
        return [e for e in self.memory.get("journal", []) 
                if datetime.fromisoformat(e["timestamp"]) >= cutoff]

    def search_journal(self, query, days=30, mood_filter=None, tag_filter=None):
        """Search journal entries with filters"""
        cutoff = datetime.now() - timedelta(days=days)
        results = []
        
        for entry in self.memory.get("journal", []):
            if datetime.fromisoformat(entry["timestamp"]) < cutoff:
                continue
            if mood_filter and entry["mood"] != mood_filter.lower():
                continue
            if tag_filter and not any(tag.lower() == tag_filter.lower() 
                                    for tag in entry.get("tags", [])):
                continue
            if query.lower() not in entry["entry"].lower():
                continue
                
            results.append(entry)
        
        return results

    def export_data(self, data_type="mood"):
        """Export data to CSV"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.export_dir}/{data_type}_export_{timestamp}.csv"
            
            if data_type == "mood":
                data = self.prepare_mood_export()
            elif data_type == "conversations":
                data = self.prepare_conversation_export()
            elif data_type == "journal":
                data = self.prepare_journal_export()
            else:
                return False, "Invalid export type"
            
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            
            return True, f"Data exported to {filename}"
        except Exception as e:
            return False, f"Export failed: {str(e)}"

    def prepare_mood_export(self):
        """Prepare mood data for export"""
        export_data = []
        
        # Daily mood data
        for day, data in self.memory.get("historical_mood", {}).items():
            export_data.append({
                "date": day,
                "type": "daily_summary",
                "positive": data["positive"],
                "neutral": data["neutral"],
                "negative": data["negative"]
            })
        
        # Mood changes
        for change in self.memory.get("daily_mood", {}).get("mood_changes", []):
            export_data.append({
                "timestamp": change["time"],
                "type": "mood_change",
                "mood": change["mood"],
                "intensity": change["intensity"]
            })
        
        return export_data

    def prepare_journal_export(self):
        """Prepare journal data for export"""
        return [{
            "timestamp": e["timestamp"],
            "mood": e["mood"],
            "tags": ", ".join(e["tags"]),
            "entry": e["entry"]
        } for e in self.memory.get("journal", [])]

    def prepare_conversation_export(self):
        """Prepare conversation history for export"""
        return [{
            "timestamp": c["time"],
            "user": c["user"],
            "response": c["response"],
            "sentiment": c.get("sentiment", ""),
            "topics": ", ".join(c.get("topics", []))
        } for c in self.memory["conversations"]]

    def load_mood_model(self):
        """Load trained mood prediction model"""
        try:
            model_path = f"{self.model_dir}/mood_predictor.pkl"
            if os.path.exists(model_path):
                self.mood_predictor = joblib.load(model_path)
        except Exception as e:
            print(f"Failed to load mood model: {e}")

    def train_mood_model(self):
        """Train mood prediction model from journal/conversation data"""
        try:
            # Prepare training data
            entries = []
            labels = []
            
            # Add journal entries
            for entry in self.memory.get("journal", []):
                entries.append(entry["entry"])
                labels.append(entry["mood"])
            
            # Add conversations with sentiment
            for conv in self.memory["conversations"]:
                if "sentiment" in conv:
                    entries.append(conv["user"])
                    labels.append(conv["sentiment"])
            
            if len(set(labels)) < 2:  # Need at least two moods to train
                return False, "Not enough diverse mood data to train"
            
            # Vectorize text
            vectorizer = TfidfVectorizer(max_features=1000)
            X = vectorizer.fit_transform(entries)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X, labels)
            
            # Save model and vectorizer
            os.makedirs(self.model_dir, exist_ok=True)
            joblib.dump({
                'model': model,
                'vectorizer': vectorizer
            }, f"{self.model_dir}/mood_predictor.pkl")
            
            self.mood_predictor = {'model': model, 'vectorizer': vectorizer}
            return True, "Mood prediction model trained successfully"
        
        except Exception as e:
            return False, f"Model training failed: {str(e)}"

    def predict_mood(self, text):
        """Predict mood from text using trained model"""
        if not self.mood_predictor:
            return None
        
        try:
            vectorized = self.mood_predictor['vectorizer'].transform([text])
            prediction = self.mood_predictor['model'].predict(vectorized)
            return prediction[0]
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def handle_journal_add(self, entry_text):
        """Process journal entry addition"""
        # Auto-detect mood if not specified
        mood_match = re.search(r"\[(positive|neutral|negative)\]", entry_text.lower())
        mood = mood_match.group(1) if mood_match else None
        
        # Auto-detect tags
        tags = []
        if "#" in entry_text:
            tags = [tag.strip("#") for tag in entry_text.split() if tag.startswith("#")]
        
        entry = self.add_journal_entry(entry_text, mood, tags)
        return f"📔 Journal entry added ({entry['mood']} mood)"

    def handle_journal_list(self, days):
        """List recent journal entries"""
        entries = self.get_journal_entries(days)
        if not entries:
            return f"No journal entries in the last {days} days"
        
        output = f"📔 Last {days} days journal entries:\n"
        for entry in entries:
            dt = datetime.fromisoformat(entry["timestamp"])
            tags = " ".join(f"#{tag}" for tag in entry["tags"]) if entry["tags"] else ""
            output += (f"\n{dt.strftime('%b %d %H:%M')} [{entry['mood']}] {tags}\n"
                     f"{entry['entry'][:100]}{'...' if len(entry['entry']) > 100 else ''}\n")
        
        return output

    def process_special_commands(self, user_input):
        """Enhanced command processing with new features"""
        lower_input = user_input.lower()
        
        # Help command
        if lower_input.startswith(("/help", "help")):
            return True, """🛠️ Nova Help Menu:
            /help - Show this menu
            /stats - Show conversation statistics
            /mood - Show today's mood summary
            /mood chart - Show mood visualizations
            /journal - Journal commands (add/list/search)
            /reminders - List upcoming reminders
            /reminders ids - Show reminders with IDs
            /modify - Change a reminder
            /delete - Remove a reminder
            /export - Export data (mood/conversations/journal/reminders)
            /import reminders - Import reminders from CSV
            /predict mood - Predict mood from text
            /train mood model - Train mood prediction model
            /forget - Clear my memory
            /topics - Show frequent topics"""
            
        # Stats command
        elif lower_input.startswith(("/stats", "stats")):
            stats = self.memory["stats"]
            return True, f"📊 Conversation Stats:\n- Total Messages: {stats['total_messages']}\n- Sessions: {stats['sessions']}\n- Known Topics: {len(self.memory['topics'])}"
            
        # Mood commands
        elif lower_input.startswith(("/mood", "mood")):
            if "chart" in lower_input:
                chart = self.generate_mood_chart('day')
                weekly_chart = self.generate_mood_chart('week')
                
                response = "📊 Mood Visualizations\n"
                if chart:
                    response += f"![Today's Mood]({chart})\n\n"
                if weekly_chart:
                    response += f"![Weekly Trend]({weekly_chart})"
                return True, response if (chart or weekly_chart) else "No enough mood data to generate charts."
            else:
                return True, self.get_mood_summary(detailed=True)
            
        # Journal commands
        elif lower_input.startswith(("/journal", "journal")):
            if "add" in lower_input:
                entry_text = user_input.split("add", 1)[1].strip()
                return True, self.handle_journal_add(entry_text)
            elif "list" in lower_input:
                days = 7
                if "last" in lower_input:
                    try:
                        days = int(re.search(r"last (\d+)", lower_input).group(1))
                    except:
                        pass
                return True, self.handle_journal_list(days)
            elif "search" in lower_input:
                parts = user_input.split(maxsplit=2)
                if len(parts) < 3:
                    return True, "Usage: /search journal [query] (optional: mood:[mood] tag:[tag] days:[num])"
                
                query = parts[2]
                mood_filter = None
                tag_filter = None
                days = 30
                
                # Parse filters
                if "mood:" in lower_input:
                    mood_filter = re.search(r"mood:(\w+)", lower_input).group(1)
                if "tag:" in lower_input:
                    tag_filter = re.search(r"tag:(\w+)", lower_input).group(1)
                if "days:" in lower_input:
                    try:
                        days = int(re.search(r"days:(\d+)", lower_input).group(1))
                    except:
                        pass
                
                results = self.search_journal(query, days, mood_filter, tag_filter)
                if not results:
                    return True, "No matching journal entries found"
                
                output = f"🔍 Journal Search Results ({len(results)} found):\n"
                for entry in results[:5]:  # Show first 5 results
                    dt = datetime.fromisoformat(entry["timestamp"])
                    tags = " ".join(f"#{tag}" for tag in entry["tags"]) if entry["tags"] else ""
                    output += (f"\n{dt.strftime('%Y-%m-%d')} [{entry['mood']}] {tags}\n"
                              f"{entry['entry'][:100]}{'...' if len(entry['entry']) > 100 else ''}\n")
                
                if len(results) > 5:
                    output += f"\n...and {len(results)-5} more entries"
                
                return True, output
            else:
                return True, """📔 Journal Help:
                /journal add [entry] - Add new journal entry
                /journal list [last X days] - List recent entries
                /journal search [query] - Search entries with filters"""
        
        # Reminder commands
        elif lower_input.startswith(("/reminders", "reminders")):
            if "ids" in lower_input:
                return True, self.list_reminders(show_ids=True)
            else:
                return True, self.list_reminders()
            
        # Reminder modification
        elif lower_input.startswith(("/modify", "modify")):
            parts = user_input.split(maxsplit=2)
            if len(parts) < 3:
                return True, "Usage: /modify [reminder_id] [new_time or 'task: new task text']"
            
            reminder_id = parts[1]
            change = parts[2]
            
            if change.startswith("task:"):
                new_task = change[5:].strip()
                return True, self.modify_reminder(reminder_id, new_task=new_task)[1]
            else:
                return True, self.modify_reminder(reminder_id, new_time=change)[1]
        
        # Reminder deletion
        elif lower_input.startswith(("/delete", "delete")):
            reminder_id = user_input.split(maxsplit=1)[1] if len(user_input.split()) > 1 else None
            if not reminder_id:
                return True, "Usage: /delete [reminder_id]"
            return True, self.delete_reminder(reminder_id)[1]
            
        # Export commands
        elif lower_input.startswith(("/export", "export")):
            if "mood" in lower_input:
                success, msg = self.export_data("mood")
            elif "conversations" in lower_input:
                success, msg = self.export_data("conversations")
            elif "journal" in lower_input:
                success, msg = self.export_data("journal")
            elif "reminders" in lower_input:
                success, msg = self.export_reminders()
            else:
                return True, """📤 Export Help:
                /export mood - Export mood data
                /export conversations - Export chat history
                /export journal - Export journal entries
                /export reminders - Export reminders"""
            return True, f"📤 {msg}"
        
        # Import commands
        elif lower_input.startswith(("/import reminders", "import reminders")):
            if len(user_input.split()) < 3:
                return True, "Usage: /import reminders [filepath]"
            filepath = user_input.split(maxsplit=2)[2]
            return True, self.import_reminders(filepath)[1]
        
        # Mood prediction commands
        elif lower_input.startswith(("/predict mood", "predict mood")):
            text_to_predict = user_input.split("predict mood", 1)[1].strip()
            if not text_to_predict:
                return True, "Usage: /predict mood [your text here]"
            
            prediction = self.predict_mood(text_to_predict)
            if prediction:
                return True, f"Predicted mood: {prediction} (based on your historical patterns)"
            else:
                return True, "Mood prediction unavailable (train a model first with /train mood model)"
        
        elif lower_input.startswith(("/train mood model", "train mood model")):
            success, message = self.train_mood_model()
            return True, f"🤖 {message}"
            
        # Forget command
        elif lower_input.startswith(("/forget", "forget")):
            self.memory = self._get_default_memory()
            self.save_memory()
            return True, "🧹 Memory cleared! I've forgotten everything."
            
        # Topics command
        elif lower_input.startswith(("/topics", "topics")):
            topics = "\n".join(f"- {t} ({c}x)" for t,c in sorted(
                self.memory["topics"].items(), key=lambda x: x[1], reverse=True)[:5])
            return True, f"📌 Frequent Topics:\n{topics or 'No topics recorded yet'}"
            
        return False, ""

    def chat(self, user_input):
        """Main chat handler with all features"""
        # Check for special commands first
        is_command, response = self.process_special_commands(user_input)
        if is_command:
            return response
            
        # Check for journal-style input
        if user_input.startswith(("Today I felt", "I'm feeling", "Right now I'm")) and len(user_input.split()) > 5:
            self.handle_journal_add(user_input)
            return "I've recorded that in your journal. Would you like to elaborate?"
            
        # Check for reminder creation
        task, time_str, frequency = self.parse_reminder(user_input)
        if task and time_str:
            success, response = self.add_reminder(task, time_str, frequency)
            if success:
                return response
        
        # Process normal input
        ai_response = self.generate_response(user_input)
        
        # Auto-detect personal info
        if re.match(r"(my name is|i'm called|you can call me) ([a-z]{2,})", user_input.lower()):
            name = re.split(r"my name is|i'm called|you can call me", user_input.lower())[-1].strip()
            self.memory["user_info"]["name"] = name
            ai_response = f"Nice to meet you, {name}! I'll remember that. 😊"
        
        # Track topics with mood context
        sentiment = self.analyze_sentiment(user_input)
        topics = self.detect_topics(user_input)
        self.track_mood(sentiment)
        
        # Store conversation with mood and topics
        self.memory["conversations"].append({
            "time": datetime.now().isoformat(),
            "user": user_input,
            "response": ai_response,
            "sentiment": sentiment,
            "topics": topics
        })
        self.memory["stats"]["total_messages"] += 1
        
        # Special responses based on mood
        if sentiment == "negative" and "sorry" not in ai_response.lower():
            ai_response += " I sense you might be feeling down. Is everything okay? 💙"
        
        # Mood prediction opportunity
        if (len(self.memory.get("journal", [])) > 10 and 
            not self.mood_predictor and
            random.random() < 0.1):  # 10% chance to suggest
            ai_response += ("\n\nI've noticed you've been journaling regularly. "
                          "Would you like me to learn your mood patterns? "
                          "Use '/train mood model' to enable mood prediction.")
        
        self.save_memory()
        return ai_response

def main():
    print("""Nova: Hi there! I'm Nova, your AI companion. 🌟
    I can track your mood, set reminders, journal your thoughts, and more.
    Type /help for special commands or just chat normally!\n""")
    
    nova = NovaCompanion()
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "bye", "quit"]:
                print("\nNova: Until next time! 👋")
                break
            
            response = nova.chat(user_input)
            print(f"Nova: {response}\n")
            
        except KeyboardInterrupt:
            print("\nNova: Oh, leaving so soon? I'll be here when you return!")
            break
        except Exception as e:
            print(f"\n⚠️ Critical error: {e}")
            print("Nova: Something went wrong! Let me reset...")
            nova = NovaCompanion()  # Fresh restart

if __name__ == "__main__":
    main()