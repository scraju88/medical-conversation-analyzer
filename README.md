# Medical Conversation Analyzer

A comprehensive medical conversation transcription and analysis system with an iOS app for recording and a Python backend for AI-powered analysis. The system can generate structured medical reports including History of Present Illness (HPI) and Assessment & Plan sections.

## Features

- **iOS App**: Native SwiftUI app for recording medical conversations
- **Python Backend**: Flask API with speech recognition and LLM analysis
- **Web Interface**: Modern web dashboard for viewing and managing conversations
- **AI Analysis**: Support for both OpenAI GPT-4 and Google Gemini
- **Database Storage**: SQLite database for conversation history
- **Real-time Processing**: Audio transcription and medical report generation

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   iOS App       │    │  Python Backend │    │  Web Interface  │
│   (SwiftUI)     │◄──►│   (Flask)       │◄──►│   (HTML/JS)     │
│                 │    │                 │    │                 │
│ • Audio Record  │    │ • Speech Recog  │    │ • View Results  │
│ • Upload Audio  │    │ • LLM Analysis  │    │ • Browse History│
│ • View Results  │    │ • Database      │    │ • Export Data   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Prerequisites

- Python 3.8+
- iOS 17.0+ (for the mobile app)
- Xcode 15.0+ (for building the iOS app)
- OpenAI API key or Google Gemini API key

## Installation

### 1. Backend Setup

1. **Clone and navigate to the project:**
   ```bash
   cd codex
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp env.example .env
   # Edit .env and add your API keys
   ```

5. **Run the backend server:**
   ```bash
   python main.py
   ```

The backend will be available at `http://localhost:5000`

### 2. iOS App Setup

1. **Open the project in Xcode:**
   ```bash
   open MedicalTranscriptionApp/MedicalTranscriptionApp.xcodeproj
   ```

2. **Configure the API endpoint:**
   - Open `ContentView.swift`
   - Update the `baseURL` in the `APIService` class to match your backend URL

3. **Build and run:**
   - Select your target device or simulator
   - Press Cmd+R to build and run

### 3. Web Interface

The web interface is automatically served by the Flask backend at `http://localhost:5000`

## Usage

### iOS App

1. **Launch the app** and grant microphone permissions
2. **Enter patient and physician information**
3. **Start recording** the medical conversation
4. **Stop recording** when finished
5. **Upload and analyze** the recording
6. **View results** including transcription and medical analysis

### Web Interface

1. **Open** `http://localhost:5000` in your browser
2. **Upload audio files** or view existing conversations
3. **Browse conversation history** with search and filtering
4. **Export results** for medical records

## API Endpoints

### POST /api/upload
Upload and process audio files.

**Parameters:**
- `audio`: Audio file (MP3, M4A, WAV)
- `patient_name`: Patient's name
- `physician_name`: Physician's name
- `llm_provider`: "openai" or "gemini"

**Response:**
```json
{
  "conversation_id": "uuid",
  "transcription": "transcribed text",
  "history_of_present_illness": "HPI analysis",
  "assessment_plan": "Assessment and plan",
  "message": "Success message"
}
```

### GET /api/conversations
Get list of all conversations.

### GET /api/conversations/{id}
Get specific conversation details.

## Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Server Configuration
FLASK_ENV=development
FLASK_DEBUG=True
```

### iOS App Configuration

Update the `baseURL` in `ContentView.swift`:

```swift
private let baseURL = "http://your-server-ip:5000"
```

## Security Considerations

- **API Keys**: Never commit API keys to version control
- **HTTPS**: Use HTTPS in production for secure data transmission
- **Authentication**: Consider adding user authentication for production use
- **Data Privacy**: Ensure compliance with HIPAA and other medical data regulations
- **Network Security**: Configure firewalls and network security appropriately

## Development

### Backend Development

The backend uses:
- **Flask**: Web framework
- **SQLAlchemy**: Database ORM
- **SpeechRecognition**: Audio transcription
- **OpenAI/Google AI**: LLM analysis

### iOS Development

The iOS app uses:
- **SwiftUI**: Modern UI framework
- **AVFoundation**: Audio recording
- **URLSession**: Network requests

## Troubleshooting

### Common Issues

1. **Microphone Permission Denied**
   - Go to Settings > Privacy > Microphone
   - Enable permission for the app

2. **API Connection Failed**
   - Check if the backend server is running
   - Verify the `baseURL` in the iOS app
   - Check network connectivity

3. **Audio Transcription Issues**
   - Ensure clear audio quality
   - Check API key configuration
   - Verify internet connection

4. **Build Errors**
   - Update Xcode to latest version
   - Clean build folder (Cmd+Shift+K)
   - Check deployment target compatibility

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This application is for educational and development purposes. For clinical use, ensure compliance with all applicable medical regulations and data privacy laws. Always consult with healthcare professionals and legal experts before deploying in a medical environment. 