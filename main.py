from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import os
import json
import datetime
import speech_recognition as sr
from pydub import AudioSegment
import openai
import google.generativeai as genai
from dotenv import load_dotenv
import tempfile
import uuid
from document_processor import DocumentProcessor

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medical_conversations.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Configure API keys
openai.api_key = os.getenv('OPENAI_API_KEY')
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Initialize document processor
document_processor = DocumentProcessor()

# Database Models
class MedicalConversation(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    patient_name = db.Column(db.String(100))
    physician_name = db.Column(db.String(100))
    conversation_date = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    audio_file_path = db.Column(db.String(500))
    transcription = db.Column(db.Text)
    history_of_present_illness = db.Column(db.Text)
    assessment_plan = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# Create database tables
with app.app_context():
    db.create_all()

def transcribe_audio(audio_file_path):
    """Transcribe audio file to text using Google Speech Recognition"""
    recognizer = sr.Recognizer()
    
    # Convert audio to WAV if needed
    audio = AudioSegment.from_file(audio_file_path)
    wav_path = audio_file_path.replace('.m4a', '.wav').replace('.mp3', '.wav')
    audio.export(wav_path, format="wav")
    
    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Error with speech recognition service: {e}"
        finally:
            # Clean up temporary WAV file
            if os.path.exists(wav_path):
                os.remove(wav_path)

def get_relevant_context(transcription):
    """Get relevant context from knowledge base"""
    try:
        # Search for relevant information based on transcription
        relevant_docs = document_processor.search_knowledge_base(transcription, top_k=3)
        
        if relevant_docs:
            context = "\n\nRELEVANT KNOWLEDGE BASE INFORMATION:\n"
            for i, doc in enumerate(relevant_docs, 1):
                context += f"{i}. {doc['content']}\n"
            return context
        else:
            return ""
    except Exception as e:
        print(f"Error getting context: {e}")
        return ""

def analyze_with_openai(transcription):
    """Analyze transcription using OpenAI GPT-4 with RAG context"""
    try:
        # Get relevant context from knowledge base
        context = get_relevant_context(transcription)
        
        prompt = f"""
        You are a medical professional analyzing a conversation between a patient and physician. 
        Please provide a structured medical report with the following sections:

        CONVERSATION TRANSCRIPT:
        {transcription}

        {context}

        Please analyze this conversation and provide:

        1. HISTORY OF PRESENT ILLNESS (HPI):
        - Chief complaint
        - History of present illness
        - Relevant symptoms and their timeline
        - Pertinent positive and negative findings

        2. ASSESSMENT AND PLAN:
        - Differential diagnosis
        - Assessment of current condition
        - Treatment plan
        - Follow-up recommendations
        - Any additional tests or referrals needed

        Use the knowledge base information to enhance your analysis with relevant medical guidelines, 
        protocols, or best practices that apply to this case.

        Format the response as a professional medical document.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical professional assistant. Provide accurate, professional medical analysis using available knowledge base information."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing with OpenAI: {str(e)}"

def analyze_with_gemini(transcription):
    """Analyze transcription using Google Gemini with RAG context"""
    try:
        model = genai.GenerativeModel('gemini-pro')
        
        # Get relevant context from knowledge base
        context = get_relevant_context(transcription)
        
        prompt = f"""
        You are a medical professional analyzing a conversation between a patient and physician. 
        Please provide a structured medical report with the following sections:

        CONVERSATION TRANSCRIPT:
        {transcription}

        {context}

        Please analyze this conversation and provide:

        1. HISTORY OF PRESENT ILLNESS (HPI):
        - Chief complaint
        - History of present illness
        - Relevant symptoms and their timeline
        - Pertinent positive and negative findings

        2. ASSESSMENT AND PLAN:
        - Differential diagnosis
        - Assessment of current condition
        - Treatment plan
        - Follow-up recommendations
        - Any additional tests or referrals needed

        Use the knowledge base information to enhance your analysis with relevant medical guidelines, 
        protocols, or best practices that apply to this case.

        Format the response as a professional medical document.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error analyzing with Gemini: {str(e)}"

@app.route('/')
def index():
    """Serve the web interface"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_audio():
    """Upload and process audio file"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        patient_name = request.form.get('patient_name', 'Unknown Patient')
        physician_name = request.form.get('physician_name', 'Unknown Physician')
        llm_provider = request.form.get('llm_provider', 'openai')  # openai or gemini
        
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Generate unique ID for the conversation
        conversation_id = str(uuid.uuid4())
        
        # Save audio file
        upload_folder = 'uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        audio_path = os.path.join(upload_folder, f"{conversation_id}_{audio_file.filename}")
        audio_file.save(audio_path)
        
        # Transcribe audio
        transcription = transcribe_audio(audio_path)
        
        # Analyze with LLM
        if llm_provider == 'gemini':
            analysis = analyze_with_gemini(transcription)
        else:
            analysis = analyze_with_openai(transcription)
        
        # Parse analysis to separate HPI and Assessment/Plan
        analysis_parts = analysis.split('\n\n')
        hpi = ""
        assessment_plan = ""
        
        for part in analysis_parts:
            if 'HISTORY OF PRESENT ILLNESS' in part.upper() or 'HPI' in part.upper():
                hpi = part
            elif 'ASSESSMENT' in part.upper() or 'PLAN' in part.upper():
                assessment_plan = part
        
        # Save to database
        conversation = MedicalConversation(
            id=conversation_id,
            patient_name=patient_name,
            physician_name=physician_name,
            audio_file_path=audio_path,
            transcription=transcription,
            history_of_present_illness=hpi,
            assessment_plan=assessment_plan
        )
        
        db.session.add(conversation)
        db.session.commit()
        
        return jsonify({
            'conversation_id': conversation_id,
            'transcription': transcription,
            'history_of_present_illness': hpi,
            'assessment_plan': assessment_plan,
            'message': 'Audio processed successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload-document', methods=['POST'])
def upload_document():
    """Upload and process a document for the knowledge base"""
    try:
        if 'document' not in request.files:
            return jsonify({'error': 'No document file provided'}), 400
        
        document_file = request.files['document']
        
        if document_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Determine file type
        filename = document_file.filename.lower()
        if filename.endswith('.pdf'):
            file_type = 'pdf'
        elif filename.endswith(('.docx', '.doc')):
            file_type = 'docx'
        elif filename.endswith('.txt'):
            file_type = 'txt'
        else:
            return jsonify({'error': 'Unsupported file type. Supported: PDF, DOCX, TXT'}), 400
        
        # Save document
        documents_folder = 'documents'
        if not os.path.exists(documents_folder):
            os.makedirs(documents_folder)
        
        document_path = os.path.join(documents_folder, document_file.filename)
        document_file.save(document_path)
        
        # Process document
        result = document_processor.process_document(document_path, file_type)
        
        if result['success']:
            return jsonify({
                'message': 'Document processed successfully',
                'chunks_created': result['chunks_created'],
                'file_type': result['file_type']
            })
        else:
            return jsonify({'error': result['error']}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/knowledge-base/stats', methods=['GET'])
def get_knowledge_base_stats():
    """Get knowledge base statistics"""
    try:
        stats = document_processor.get_knowledge_base_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/knowledge-base/clear', methods=['POST'])
def clear_knowledge_base():
    """Clear the knowledge base"""
    try:
        result = document_processor.clear_knowledge_base()
        if result['success']:
            return jsonify({'message': result['message']})
        else:
            return jsonify({'error': result['error']}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    """Get all conversations"""
    try:
        conversations = MedicalConversation.query.order_by(MedicalConversation.created_at.desc()).all()
        result = []
        
        for conv in conversations:
            result.append({
                'id': conv.id,
                'patient_name': conv.patient_name,
                'physician_name': conv.physician_name,
                'conversation_date': conv.conversation_date.isoformat(),
                'created_at': conv.created_at.isoformat()
            })
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversations/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    """Get specific conversation details"""
    try:
        conversation = MedicalConversation.query.get(conversation_id)
        if not conversation:
            return jsonify({'error': 'Conversation not found'}), 404
        
        return jsonify({
            'id': conversation.id,
            'patient_name': conversation.patient_name,
            'physician_name': conversation.physician_name,
            'conversation_date': conversation.conversation_date.isoformat(),
            'transcription': conversation.transcription,
            'history_of_present_illness': conversation.history_of_present_illness,
            'assessment_plan': conversation.assessment_plan,
            'created_at': conversation.created_at.isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
