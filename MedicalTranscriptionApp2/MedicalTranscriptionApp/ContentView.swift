import SwiftUI
import AVFoundation
import AVKit

struct ContentView: View {
    @StateObject private var audioRecorder = AudioRecorder()
    @StateObject private var apiService = APIService()
    
    @State private var patientName = ""
    @State private var physicianName = ""
    @State private var selectedLLMProvider = "openai"
    @State private var showingResults = false
    @State private var showingAlert = false
    @State private var alertMessage = ""
    @State private var isUploading = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Header
                    VStack {
                        Image(systemName: "stethoscope")
                            .font(.system(size: 60))
                            .foregroundColor(.blue)
                        
                        Text("Medical Conversation Analyzer")
                            .font(.largeTitle)
                            .fontWeight(.bold)
                            .multilineTextAlignment(.center)
                        
                        Text("Record medical conversations and get AI-powered analysis")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                    }
                    .padding()
                    
                    // Recording Section
                    VStack(spacing: 15) {
                        HStack {
                            Image(systemName: "mic.fill")
                                .foregroundColor(.blue)
                            Text("Audio Recording")
                                .font(.headline)
                            Spacer()
                        }
                        
                        // Recording Button
                        Button(action: {
                            if audioRecorder.isRecording {
                                audioRecorder.stopRecording()
                            } else {
                                audioRecorder.startRecording()
                            }
                        }) {
                            HStack {
                                Image(systemName: audioRecorder.isRecording ? "stop.fill" : "mic.fill")
                                    .font(.title2)
                                Text(audioRecorder.isRecording ? "Stop Recording" : "Start Recording")
                                    .fontWeight(.semibold)
                            }
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(audioRecorder.isRecording ? Color.red : Color.blue)
                            .cornerRadius(12)
                        }
                        .disabled(audioRecorder.isRecording)
                        
                        // Recording Status
                        if audioRecorder.isRecording {
                            HStack {
                                Image(systemName: "record.circle")
                                    .foregroundColor(.red)
                                    .scaleEffect(1.2)
                                Text("Recording...")
                                    .foregroundColor(.red)
                                    .fontWeight(.medium)
                            }
                        }
                        
                        // Recording Duration
                        if audioRecorder.recordingDuration > 0 {
                            Text(formatDuration(audioRecorder.recordingDuration))
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                    
                    // Form Section
                    VStack(spacing: 15) {
                        HStack {
                            Image(systemName: "person.fill")
                                .foregroundColor(.blue)
                            Text("Patient Information")
                                .font(.headline)
                            Spacer()
                        }
                        
                        VStack(spacing: 12) {
                            TextField("Patient Name", text: $patientName)
                                .textFieldStyle(RoundedBorderTextFieldStyle())
                            
                            TextField("Physician Name", text: $physicianName)
                                .textFieldStyle(RoundedBorderTextFieldStyle())
                            
                            Picker("AI Provider", selection: $selectedLLMProvider) {
                                Text("OpenAI GPT-4").tag("openai")
                                Text("Google Gemini").tag("gemini")
                            }
                            .pickerStyle(SegmentedPickerStyle())
                        }
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                    
                    // Upload Button
                    Button(action: uploadRecording) {
                        HStack {
                            if isUploading {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                    .scaleEffect(0.8)
                            } else {
                                Image(systemName: "arrow.up.circle.fill")
                                    .font(.title2)
                            }
                            Text(isUploading ? "Processing..." : "Upload & Analyze")
                                .fontWeight(.semibold)
                        }
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(audioRecorder.audioURL != nil ? Color.blue : Color.gray)
                        .cornerRadius(12)
                    }
                    .disabled(audioRecorder.audioURL == nil || patientName.isEmpty || physicianName.isEmpty || isUploading)
                    
                    // Results Section
                    if showingResults {
                        VStack(spacing: 15) {
                            HStack {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.green)
                                Text("Analysis Complete")
                                    .font(.headline)
                                Spacer()
                            }
                            
                            if let results = apiService.lastResults {
                                VStack(spacing: 12) {
                                    ResultSection(title: "History of Present Illness", content: results.historyOfPresentIllness)
                                    ResultSection(title: "Assessment & Plan", content: results.assessmentPlan)
                                    ResultSection(title: "Transcription", content: results.transcription)
                                }
                            }
                        }
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(12)
                    }
                    
                    Spacer(minLength: 50)
                }
                .padding()
            }
            .navigationBarHidden(true)
        }
        .alert("Message", isPresented: $showingAlert) {
            Button("OK") { }
        } message: {
            Text(alertMessage)
        }
        .onReceive(audioRecorder.$recordingDuration) { _ in
            // Update UI when recording duration changes
        }
    }
    
    private func uploadRecording() {
        guard let audioURL = audioRecorder.audioURL else {
            showAlert("No recording available")
            return
        }
        
        isUploading = true
        
        apiService.uploadAudio(
            audioURL: audioURL,
            patientName: patientName,
            physicianName: physicianName,
            llmProvider: selectedLLMProvider
        ) { success, message in
            DispatchQueue.main.async {
                isUploading = false
                
                if success {
                    showingResults = true
                    showAlert("Analysis completed successfully!")
                } else {
                    showAlert("Error: \(message)")
                }
            }
        }
    }
    
    private func showAlert(_ message: String) {
        alertMessage = message
        showingAlert = true
    }
    
    private func formatDuration(_ duration: TimeInterval) -> String {
        let minutes = Int(duration) / 60
        let seconds = Int(duration) % 60
        return String(format: "%02d:%02d", minutes, seconds)
    }
}

struct ResultSection: View {
    let title: String
    let content: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
                .foregroundColor(.blue)
            
            Text(content.isEmpty ? "No content available" : content)
                .font(.body)
                .foregroundColor(.primary)
                .padding()
                .background(Color(.systemBackground))
                .cornerRadius(8)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

// Audio Recorder
class AudioRecorder: NSObject, ObservableObject {
    @Published var isRecording = false
    @Published var recordingDuration: TimeInterval = 0
    
    private var audioRecorder: AVAudioRecorder?
    private var timer: Timer?
    private var startTime: Date?
    
    var audioURL: URL? {
        return audioRecorder?.url
    }
    
    override init() {
        super.init()
        setupAudioSession()
    }
    
    private func setupAudioSession() {
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.playAndRecord, mode: .default)
            try audioSession.setActive(true)
        } catch {
            print("Failed to setup audio session: \(error)")
        }
    }
    
    func startRecording() {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let audioFilename = documentsPath.appendingPathComponent("recording.m4a")
        
        let settings = [
            AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
            AVSampleRateKey: 12000,
            AVNumberOfChannelsKey: 1,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
        ]
        
        do {
            audioRecorder = try AVAudioRecorder(url: audioFilename, settings: settings)
            audioRecorder?.delegate = self
            audioRecorder?.record()
            
            isRecording = true
            startTime = Date()
            
            timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
                self.recordingDuration = Date().timeIntervalSince(self.startTime ?? Date())
            }
        } catch {
            print("Could not start recording: \(error)")
        }
    }
    
    func stopRecording() {
        audioRecorder?.stop()
        timer?.invalidate()
        timer = nil
        
        isRecording = false
    }
}

extension AudioRecorder: AVAudioRecorderDelegate {
    func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        if !flag {
            print("Recording failed")
        }
    }
}

// API Service
class APIService: ObservableObject {
    @Published var lastResults: AnalysisResults?
    
    private let baseURL = "http://localhost:5000" // Change to your server URL
    
    struct AnalysisResults {
        let historyOfPresentIllness: String
        let assessmentPlan: String
        let transcription: String
    }
    
    func uploadAudio(audioURL: URL, patientName: String, physicianName: String, llmProvider: String, completion: @escaping (Bool, String) -> Void) {
        let url = URL(string: "\(baseURL)/api/upload")!
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        
        // Add form fields
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"patient_name\"\r\n\r\n".data(using: .utf8)!)
        body.append("\(patientName)\r\n".data(using: .utf8)!)
        
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"physician_name\"\r\n\r\n".data(using: .utf8)!)
        body.append("\(physicianName)\r\n".data(using: .utf8)!)
        
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"llm_provider\"\r\n\r\n".data(using: .utf8)!)
        body.append("\(llmProvider)\r\n".data(using: .utf8)!)
        
        // Add audio file
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"audio\"; filename=\"recording.m4a\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: audio/m4a\r\n\r\n".data(using: .utf8)!)
        
        do {
            let audioData = try Data(contentsOf: audioURL)
            body.append(audioData)
        } catch {
            completion(false, "Failed to read audio file")
            return
        }
        
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    completion(false, error.localizedDescription)
                    return
                }
                
                guard let data = data else {
                    completion(false, "No data received")
                    return
                }
                
                do {
                    let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                    
                    if let error = json?["error"] as? String {
                        completion(false, error)
                        return
                    }
                    
                    if let transcription = json?["transcription"] as? String,
                       let hpi = json?["history_of_present_illness"] as? String,
                       let assessment = json?["assessment_plan"] as? String {
                        
                        self.lastResults = AnalysisResults(
                            historyOfPresentIllness: hpi,
                            assessmentPlan: assessment,
                            transcription: transcription
                        )
                        
                        completion(true, "Success")
                    } else {
                        completion(false, "Invalid response format")
                    }
                } catch {
                    completion(false, "Failed to parse response")
                }
            }
        }.resume()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
} 