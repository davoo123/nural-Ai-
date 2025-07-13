#!/usr/bin/env python3
"""
Setup script for Neural AI Assistant
"""

import os
import json
import shutil

def setup_ai():
    """Initialize the AI assistant for first use."""
    print("🤖 Setting up Neural AI Assistant...")
    
    # Create knowledge_base directory if it doesn't exist
    if not os.path.exists('knowledge_base'):
        os.makedirs('knowledge_base')
        print("✅ Created knowledge_base directory")
    
    # Create memory.json if it doesn't exist
    memory_file = 'knowledge_base/memory.json'
    if not os.path.exists(memory_file):
        # Copy sample memory file
        if os.path.exists('knowledge_base/memory_sample.json'):
            shutil.copy('knowledge_base/memory_sample.json', memory_file)
            print("✅ Created memory.json from sample")
        else:
            # Create basic memory structure
            basic_memory = {
                "facts": [],
                "questions": {}
            }
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(basic_memory, f, indent=2)
            print("✅ Created empty memory.json")
    
    # Create user profile if it doesn't exist
    profile_file = 'knowledge_base/user_profile.json'
    if not os.path.exists(profile_file):
        basic_profile = {
            "name": "User",
            "preferences": {},
            "interaction_history": []
        }
        with open(profile_file, 'w', encoding='utf-8') as f:
            json.dump(basic_profile, f, indent=2)
        print("✅ Created user_profile.json")
    
    print("\n🎉 Setup complete!")
    print("\n🚀 To start your AI assistant:")
    print("   python ai_assistant.py")
    print("\n💡 Useful commands:")
    print("   learn mode          - Start autonomous learning")
    print("   consciousness report - View AI consciousness state")
    print("   who are you         - AI self-description")
    print("   help               - Show all commands")

if __name__ == "__main__":
    setup_ai()
