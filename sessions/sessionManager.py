"""
Session Management System for RAG Chatbot
Handles user sessions, conversation history, and session persistence
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import hashlib


class SessionManager:
    """Manages user sessions and conversation history"""
    
    def __init__(self, sessions_dir: str = "sessions"):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(exist_ok=True)
        self.current_session: Optional[str] = None
        self.sessions: Dict[str, Dict] = {}
        self._load_all_sessions()
    
    def _load_all_sessions(self):
        """Load all existing sessions from disk"""
        for session_file in self.sessions_dir.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                    session_id = session_file.stem
                    self.sessions[session_id] = session_data
            except Exception as e:
                print(f"Error loading session {session_file}: {e}")
    
    def create_session(self, user_name: Optional[str] = None) -> str:
        """Create a new session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if user_name:
            # Create session ID from username and timestamp
            session_id = f"{user_name}_{timestamp}"
        else:
            # Create anonymous session
            session_id = f"anonymous_{timestamp}"
        
        session_data = {
            "session_id": session_id,
            "user_name": user_name or "Anonymous",
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "message_count": 0,
            "history": []
        }
        
        self.sessions[session_id] = session_data
        self.current_session = session_id
        self._save_session(session_id)
        
        return session_id
    
    def load_session(self, session_id: str) -> bool:
        """Load an existing session"""
        if session_id in self.sessions:
            self.current_session = session_id
            # Update last active time
            self.sessions[session_id]["last_active"] = datetime.now().isoformat()
            self._save_session(session_id)
            return True
        return False
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data"""
        return self.sessions.get(session_id)
    
    def list_sessions(self, user_name: Optional[str] = None) -> List[Dict]:
        """List all sessions, optionally filtered by username"""
        sessions = []
        for session_id, session_data in self.sessions.items():
            if user_name is None or session_data.get("user_name") == user_name:
                sessions.append({
                    "session_id": session_id,
                    "user_name": session_data.get("user_name"),
                    "created_at": session_data.get("created_at"),
                    "last_active": session_data.get("last_active"),
                    "message_count": session_data.get("message_count", 0)
                })
        
        # Sort by last active (most recent first)
        sessions.sort(key=lambda x: x["last_active"], reverse=True)
        return sessions
    
    def add_message(self, role: str, content: str, session_id: Optional[str] = None):
        """Add a message to session history"""
        sid = session_id or self.current_session
        if not sid or sid not in self.sessions:
            raise ValueError("No active session")
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        self.sessions[sid]["history"].append(message)
        self.sessions[sid]["message_count"] = len(self.sessions[sid]["history"])
        self.sessions[sid]["last_active"] = datetime.now().isoformat()
        self._save_session(sid)
    
    def get_history(self, session_id: Optional[str] = None, last_n: Optional[int] = None) -> List[Dict]:
        """Get conversation history for a session"""
        sid = session_id or self.current_session
        if not sid or sid not in self.sessions:
            return []
        
        history = self.sessions[sid]["history"]
        if last_n:
            return history[-last_n:]
        return history
    
    def clear_history(self, session_id: Optional[str] = None):
        """Clear conversation history for a session"""
        sid = session_id or self.current_session
        if sid and sid in self.sessions:
            self.sessions[sid]["history"] = []
            self.sessions[sid]["message_count"] = 0
            self._save_session(sid)
            print(f"🗑️  History cleared for session: {sid}")
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session permanently"""
        if session_id in self.sessions:
            # Remove from memory
            del self.sessions[session_id]
            
            # Remove from disk
            session_file = self.sessions_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
            
            # Clear current session if it was deleted
            if self.current_session == session_id:
                self.current_session = None
            
            print(f"🗑️  Session deleted: {session_id}")
            return True
        return False
    
    def get_session_summary(self, session_id: Optional[str] = None) -> str:
        """Get a summary of the session"""
        sid = session_id or self.current_session
        if not sid or sid not in self.sessions:
            return "No active session"
        
        session = self.sessions[sid]
        return (f"Session: {session['user_name']} | "
                f"Messages: {session['message_count']} | "
                f"Created: {session['created_at'][:10]}")
    
    def display_session_info(self, session_id: Optional[str] = None):
        """Display detailed session information"""
        sid = session_id or self.current_session
        if not sid or sid not in self.sessions:
            print("No active session")
            return
        
        session = self.sessions[sid]
        print("\n" + "="*60)
        print("SESSION INFORMATION")
        print("="*60)
        print(f"Session ID: {session['session_id']}")
        print(f"User: {session['user_name']}")
        print(f"Created: {session['created_at']}")
        print(f"Last Active: {session['last_active']}")
        print(f"Total Messages: {session['message_count']}")
        print("="*60 + "\n")
    
    def display_history(self, last_n: int = 5, session_id: Optional[str] = None):
        """Display recent conversation history"""
        history = self.get_history(session_id, last_n)
        
        print("\n" + "="*60)
        print(f"RECENT CONVERSATION (Last {len(history)} messages)")
        print("="*60)
        for msg in history:
            timestamp = msg.get('timestamp', '')[:19]  # Show date and time
            print(f"[{timestamp}] {msg['role'].upper()}: {msg['content'][:100]}...")
        print("="*60 + "\n")
    
    def _save_session(self, session_id: str):
        """Save session to disk (internal)"""
        if session_id not in self.sessions:
            return
        session_file = self.sessions_dir / f"{session_id}.json"
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(self.sessions[session_id], f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving session {session_id}: {e}")

    def save_session(self, session_id: Optional[str] = None) -> bool:
        """Public API: save a session to disk."""
        sid = session_id or self.current_session
        if not sid or sid not in self.sessions:
            return False
        self._save_session(sid)
        return True
    
    def export_session(self, session_id: Optional[str] = None, export_path: Optional[str] = None) -> str:
        """Export session history to a text file"""
        sid = session_id or self.current_session
        if not sid or sid not in self.sessions:
            raise ValueError("No active session")
        
        session = self.sessions[sid]
        
        if not export_path:
            export_path = f"session_export_{sid}.txt"
        
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(f"Chat Session Export\n")
            f.write(f"="*60 + "\n")
            f.write(f"Session ID: {session['session_id']}\n")
            f.write(f"User: {session['user_name']}\n")
            f.write(f"Created: {session['created_at']}\n")
            f.write(f"Messages: {session['message_count']}\n")
            f.write(f"="*60 + "\n\n")
            
            for msg in session['history']:
                timestamp = msg.get('timestamp', '')
                f.write(f"[{timestamp}] {msg['role'].upper()}:\n")
                f.write(f"{msg['content']}\n\n")
        
        print(f"✅ Session exported to: {export_path}")
        return export_path


class SessionSelector:
    """Interactive session selector for CLI"""
    
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
    
    def select_or_create_session(self) -> str:
        """Interactive session selection"""
        print("\n" + "="*60)
        print("SESSION MANAGEMENT")
        print("="*60)
        
        # List existing sessions
        sessions = self.session_manager.list_sessions()
        
        if sessions:
            print("\nExisting Sessions:")
            for i, session in enumerate(sessions[:10], 1):  # Show last 10 sessions
                created = session['created_at'][:10]
                active = session['last_active'][:10]
                print(f"{i}. {session['user_name']} - {session['message_count']} messages "
                      f"(Created: {created}, Active: {active})")
            print(f"\n0. Create new session")
        else:
            print("\nNo existing sessions found.")
            print("Let's create a new session!")
        
        print("="*60)
        
        # Get user choice
        while True:
            if sessions:
                choice = input("\nSelect session number (or 0 for new): ").strip()
                
                if choice == '0':
                    return self._create_new_session()
                
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(sessions):
                        session_id = sessions[idx]['session_id']
                        self.session_manager.load_session(session_id)
                        print(f"\n✅ Loaded session: {sessions[idx]['user_name']}")
                        return session_id
                    else:
                        print("Invalid selection. Try again.")
                except ValueError:
                    print("Invalid input. Enter a number.")
            else:
                return self._create_new_session()
    
    def _create_new_session(self) -> str:
        """Create a new session interactively"""
        print("\n" + "-"*60)
        print("CREATE NEW SESSION")
        print("-"*60)
        
        user_name = input("Enter your name (or press Enter for anonymous): ").strip()
        
        if not user_name:
            user_name = None
        
        session_id = self.session_manager.create_session(user_name)
        print(f"\n✅ New session created: {session_id}")
        
        return session_id