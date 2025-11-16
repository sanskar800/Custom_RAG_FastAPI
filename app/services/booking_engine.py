import re
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone
from pymongo import MongoClient
from groq import Groq
from app.config import settings

class BookingEngine:
    """State machine for handling interview booking flow."""
    
    def __init__(self):
        self.mongo_client = MongoClient(settings.MONGODB_URI)
        self.db = self.mongo_client[settings.MONGO_DB_NAME]
        self.bookings_collection = self.db[settings.MONGO_BOOKINGS_COLLECTION]
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
    
    def detect_booking_intent(self, user_text: str) -> bool:
        """
        Detect if user wants to book an interview.
        
        Args:
            user_text: User's message
            
        Returns:
            True if booking intent detected
        """
        # Keyword-based detection
        keywords = {
            "book", "booking", "interview", "schedule", 
            "appointment", "slot", "meet", "meeting"
        }
        text_lower = user_text.lower()
        keyword_match = any(keyword in text_lower for keyword in keywords)
        
        # LLM-based intent classification
        try:
            system_prompt = (
                "You are an intent classifier. Analyze the user's message and determine "
                "if they want to book an interview or schedule an appointment. "
                "Respond with ONLY one word: BOOKING or OTHER"
            )
            
            response = self.groq_client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User message: \"{user_text}\""}
                ],
                temperature=0.0,
                max_tokens=10
            )
            
            classification = response.choices[0].message.content.strip().upper()
            llm_match = "BOOK" in classification
            
            return llm_match or keyword_match
        except Exception:
            return keyword_match
    
    def start_booking(self) -> str:
        """
        Start the booking process.
        
        Returns:
            Initial booking prompt
        """
        return (
            "I'll help you schedule an interview. Let's get started!\n\n"
            "Please provide your **full name**."
        )
    
    def process_booking_input(
        self, 
        session_id: str, 
        user_input: str, 
        current_state: Dict
    ) -> Tuple[str, Optional[Dict], bool]:
        """
        Process user input in the booking flow.
        
        Args:
            session_id: Session identifier
            user_input: User's input
            current_state: Current booking state
            
        Returns:
            Tuple of (response_message, completed_booking_data, is_complete)
        """
        step = current_state.get("step", "name")
        
        if step == "name":
            return self._process_name(user_input, current_state)
        elif step == "email":
            return self._process_email(user_input, current_state)
        elif step == "date":
            return self._process_date(user_input, current_state)
        elif step == "time":
            return self._process_time(user_input, current_state)
        
        return "Invalid booking state. Please start over.", None, False
    
    def _process_name(
        self, 
        user_input: str, 
        state: Dict
    ) -> Tuple[str, Optional[Dict], bool]:
        """Process name input."""
        name = user_input.strip()
        
        if len(name) < 2:
            return "Please provide a valid full name (at least 2 characters).", None, False
        
        state["name"] = name
        state["step"] = "email"
        
        return f"Thank you, {name}! Now, please provide your **email address**.", None, False
    
    def _process_email(
        self, 
        user_input: str, 
        state: Dict
    ) -> Tuple[str, Optional[Dict], bool]:
        """Process email input."""
        email = user_input.strip()
        
        # Validate email format
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            return (
                "That doesn't look like a valid email address. "
                "Please provide a valid email (e.g., user@example.com).",
                None,
                False
            )
        
        state["email"] = email
        state["step"] = "date"
        
        return (
            "Great! Now, please provide your preferred **date** for the interview.\n"
            "Format: YYYY-MM-DD (e.g., 2025-11-20)",
            None,
            False
        )
    
    def _process_date(
        self, 
        user_input: str, 
        state: Dict
    ) -> Tuple[str, Optional[Dict], bool]:
        """Process date input."""
        date_str = user_input.strip()
        
        # Validate date format
        date_pattern = r'^\d{4}-\d{2}-\d{2}$'
        if not re.match(date_pattern, date_str):
            return (
                "Invalid date format. Please use YYYY-MM-DD format (e.g., 2025-11-20).",
                None,
                False
            )
        
        # Validate date is valid and in the future
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            if date_obj.date() < datetime.now().date():
                return "Please provide a future date.", None, False
        except ValueError:
            return "Invalid date. Please provide a valid date in YYYY-MM-DD format.", None, False
        
        state["date"] = date_str
        state["step"] = "time"
        
        return (
            "Perfect! Finally, please provide your preferred **time** for the interview.\n"
            "Format: HH:MM in 24-hour format (e.g., 14:30 for 2:30 PM)",
            None,
            False
        )
    
    def _process_time(
        self, 
        user_input: str, 
        state: Dict
    ) -> Tuple[str, Optional[Dict], bool]:
        """Process time input and complete booking."""
        time_str = user_input.strip()
        
        # Validate time format
        time_pattern = r'^\d{2}:\d{2}$'
        if not re.match(time_pattern, time_str):
            return (
                "Invalid time format. Please use HH:MM format in 24-hour notation (e.g., 14:30).",
                None,
                False
            )
        
        # Validate time is valid
        try:
            time_obj = datetime.strptime(time_str, "%H:%M")
        except ValueError:
            return "Invalid time. Please provide a valid time in HH:MM format.", None, False
        
        state["time"] = time_str
        
        # Save booking to database
        booking_data = self._save_booking(state)
        
        response = (
            "🎉 **Booking Confirmed!**\n\n"
            f"**Booking ID:** {booking_data['booking_id']}\n"
            f"**Name:** {booking_data['name']}\n"
            f"**Email:** {booking_data['email']}\n"
            f"**Date:** {booking_data['date']}\n"
            f"**Time:** {booking_data['time']}\n\n"
            "You will receive a confirmation email shortly. Looking forward to meeting you!"
        )
        
        return response, booking_data, True
    
    def _save_booking(self, booking_info: Dict) -> Dict:
        """
        Save booking to MongoDB.
        
        Args:
            booking_info: Booking information
            
        Returns:
            Saved booking record
        """
        try:
            now = datetime.now(timezone.utc)
            booking_id = f"BOOK_{now.strftime('%Y%m%d%H%M%S%f')}"
            
            booking_record = {
                "booking_id": booking_id,
                "name": booking_info["name"],
                "email": booking_info["email"],
                "date": booking_info["date"],
                "time": booking_info["time"],
                "status": "confirmed",
                "created_at": now
            }
            
            self.bookings_collection.insert_one(booking_record)
            
            # Return serializable version
            return {
                "booking_id": booking_id,
                "name": booking_info["name"],
                "email": booking_info["email"],
                "date": booking_info["date"],
                "time": booking_info["time"],
                "status": "confirmed",
                "created_at": now.isoformat()
            }
        except Exception as e:
            raise RuntimeError(f"Failed to save booking: {str(e)}")

# Singleton instance
booking_engine = BookingEngine()