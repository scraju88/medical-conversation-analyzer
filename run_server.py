#!/usr/bin/env python3
"""
Medical Conversation Analyzer - Server Startup Script
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        import openai
        import google.generativeai
        import speech_recognition
        import pydub
        print("✓ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_env_file():
    """Check if .env file exists and has required keys"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found")
        print("Please copy env.example to .env and add your API keys")
        return False
    
    with open(env_file) as f:
        content = f.read()
        if "your_openai_api_key_here" in content or "your_gemini_api_key_here" in content:
            print("⚠️  Please update your API keys in .env file")
            return False
    
    print("✓ Environment variables configured")
    return True

def create_upload_directory():
    """Create uploads directory if it doesn't exist"""
    upload_dir = Path("uploads")
    if not upload_dir.exists():
        upload_dir.mkdir()
        print("✓ Created uploads directory")

def main():
    """Main startup function"""
    print("🏥 Medical Conversation Analyzer - Server Startup")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment
    if not check_env_file():
        print("\nTo continue without API keys (for testing), press Enter...")
        input()
    
    # Create necessary directories
    create_upload_directory()
    
    # Import and run the Flask app
    try:
        from main import app
        print("\n🚀 Starting server...")
        print("📱 iOS App: Configure baseURL to http://localhost:5100")
        print("🌐 Web Interface: http://localhost:5100")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 50)
        
        app.run(debug=True, host='0.0.0.0', port=5100)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 