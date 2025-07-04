#!/usr/bin/env python3
"""
Test script for Medical Conversation Analyzer Backend
"""

import requests
import json
import time
from pathlib import Path

def test_server_health():
    """Test if the server is running"""
    try:
        response = requests.get("http://localhost:5100/")
        if response.status_code == 200:
            print("✓ Server is running")
            return True
        else:
            print(f"✗ Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server. Is it running?")
        return False

def test_conversations_endpoint():
    """Test the conversations endpoint"""
    try:
        response = requests.get("http://localhost:5100/api/conversations")
        if response.status_code == 200:
            conversations = response.json()
            print(f"✓ Conversations endpoint working (found {len(conversations)} conversations)")
            return True
        else:
            print(f"✗ Conversations endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error testing conversations endpoint: {e}")
        return False

def test_upload_without_audio():
    """Test upload endpoint without audio file"""
    try:
        data = {
            'patient_name': 'Test Patient',
            'physician_name': 'Test Doctor',
            'llm_provider': 'openai'
        }
        
        response = requests.post("http://localhost:5100/api/upload", data=data)
        if response.status_code == 400:
            print("✓ Upload endpoint correctly rejects request without audio")
            return True
        else:
            print(f"✗ Upload endpoint unexpected response: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error testing upload endpoint: {e}")
        return False

def create_test_audio():
    """Create a simple test audio file"""
    try:
        # This is a minimal WAV file header for testing
        wav_header = (
            b'RIFF' + (36).to_bytes(4, 'little') + b'WAVE' +
            b'fmt ' + (16).to_bytes(4, 'little') + (1).to_bytes(2, 'little') +
            (1).to_bytes(2, 'little') + (8000).to_bytes(4, 'little') +
            (8000).to_bytes(4, 'little') + (1).to_bytes(2, 'little') +
            (8).to_bytes(2, 'little') + b'data' + (0).to_bytes(4, 'little')
        )
        
        test_file = Path("test_audio.wav")
        with open(test_file, 'wb') as f:
            f.write(wav_header)
        
        print("✓ Created test audio file")
        return test_file
    except Exception as e:
        print(f"✗ Error creating test audio: {e}")
        return None

def test_upload_with_test_audio():
    """Test upload with a minimal test audio file"""
    test_file = create_test_audio()
    if not test_file:
        return False
    
    try:
        with open(test_file, 'rb') as f:
            files = {'audio': ('test.wav', f, 'audio/wav')}
            data = {
                'patient_name': 'Test Patient',
                'physician_name': 'Test Doctor',
                'llm_provider': 'openai'
            }
            
            response = requests.post("http://localhost:5100/api/upload", 
                                   files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print("✓ Upload with test audio successful")
                print(f"  - Conversation ID: {result.get('conversation_id', 'N/A')}")
                print(f"  - Transcription: {result.get('transcription', 'N/A')[:50]}...")
                return True
            else:
                print(f"✗ Upload failed: {response.status_code}")
                print(f"  Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"✗ Error testing upload with audio: {e}")
        return False
    finally:
        # Clean up test file
        if test_file and test_file.exists():
            test_file.unlink()

def main():
    """Run all tests"""
    print("🧪 Medical Conversation Analyzer - Backend Tests")
    print("=" * 50)
    
    tests = [
        ("Server Health", test_server_health),
        ("Conversations Endpoint", test_conversations_endpoint),
        ("Upload Validation", test_upload_without_audio),
        ("Upload with Test Audio", test_upload_with_test_audio),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Backend is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the server configuration.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 