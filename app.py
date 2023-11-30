from vosk import Model, KaldiRecognizer
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import wave
import json
from pydub import AudioSegment
from fun import analyze_audio, compare_lines, count_duplicate_lines, count_skipped_lines, count_words,calculate_word_count_ratio,get_wav_duration

app = Flask(__name__)
CORS(app)

model = Model("/home/alicode/Downloads/vosk-model-small-nl-0.22")



def calculate_error_metrics(original_text, transcribed_text):
    """
    Calculate error metrics between original and transcribed text.

    Parameters:
    - original_text (str): Original text.
    - transcribed_text (str): Transcribed text.

    Returns:
    - dict: Error metrics.
    """
    words_original = original_text.split()
    words_transcribed = transcribed_text.split()

    wc = len(set(words_original) & set(words_transcribed))  # Words Correct
    wr = len(words_transcribed)  # Words Reads

    # Implement your logic to calculate other error metrics
    # ...

    error_metrics = {
        'WR': wr,
        'WC': wc,
        'Words Correct per Minute': calculate_words_per_minute(wc, transcribed_text),
        # Add other error metrics as needed
    }

    return error_metrics

def calculate_words_per_minute(words_correct, transcribed_text):
    """
    Calculate words correct per minute.

    Parameters:
    - words_correct (int): Number of words correct.
    - transcribed_text (str): Transcribed text.

    Returns:
    - float: Words correct per minute.
    """
    # Implement your logic to calculate words correct per minute
    # ...

    # Placeholder value, replace with actual calculation
    audio_duration = 120  # seconds
    words_per_minute = (words_correct / audio_duration) * 60

    return words_per_minute

def calculate_pause_metrics(transcribed_text):
    """
    Calculate pause metrics in the transcribed text.

    Parameters:
    - transcribed_text (str): Transcribed text.

    Returns:
    - dict: Pause metrics.
    """
    # Implement your logic to calculate pause metrics
    # ...

    # Placeholder values, replace with actual calculations
    pauses_1_3_seconds = 5
    hesitations_3_seconds = 2

    pause_metrics = {
        'Pauses (1-3 seconds)': pauses_1_3_seconds,
        'Hesitations (3+ seconds)': hesitations_3_seconds,
        # Add other pause metrics as needed
    }

    return pause_metrics
def transcribe_audio_vosk(audio_file_path):
    wf = wave.open(audio_file_path, "rb")
    recognizer = KaldiRecognizer(model, wf.getframerate())
    
    accumulated_text = ""
    
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        recognizer.AcceptWaveform(data)
    
    result = json.loads(recognizer.FinalResult())
    accumulated_text += result.get("text", "")
    
    return accumulated_text

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        data = request.get_json()
        audio_url = data.get('audio_url')
        original_text = data.get('original_text')

        # Download the audio file from the provided URL
        response = requests.get(audio_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download audio from the provided URL. Status code: {response.status_code}")

        # Save audio data to a temporary file
        audio_file_path = "temp_original.wav"
        with open(audio_file_path, "wb") as temp_file:
            temp_file.write(response.content)

        # Convert audio to mono and set sample width to 2 bytes
        audio = AudioSegment.from_file(audio_file_path)
        audio = audio.set_channels(1).set_sample_width(2)
        audio.export("temp.wav", format="wav")

        # Transcribe audio using Vosk
        transcribed_text = transcribe_audio_vosk("temp.wav")
        deleted_words, inserted_words, substituted_words, repeated_words = compare_lines(original_text, transcribed_text)
        duplicate_lines = count_duplicate_lines(transcribed_text)
        skipped_lines = count_skipped_lines(transcribed_text)
        word_count = count_words(transcribed_text)


    # Calculate error metrics
        error_metrics = calculate_error_metrics(original_text, transcribed_text)

        # Calculate pause metrics
        pause_metrics = calculate_pause_metrics(transcribed_text)
        # Additional analysis and metrics calculations go here...

        # Calculate accuracy, audio duration, and transcription confidence score
        accuracy = error_metrics.get('Words Correct per Minute', 0) / word_count * 100 if word_count != 0 else 0
        audio_duration = get_wav_duration('temp.wav')
        # transcription_confidence = confidence if confidence is not None else 0

        original_vs_audio = calculate_word_count_ratio(transcribed_text, original_text)
        analysis_result = analyze_audio('temp.wav')
        # Prepare JSON response with additional outcomes
        response = {
            'transcribed_text': transcribed_text,
            # 'confidence': confidence,  # Confidence score
            'analysis_result': analysis_result,
            'deleted_words': deleted_words,
            'inserted_words': inserted_words,
            'substituted_words': substituted_words,
            'repeated_words': repeated_words,
            'duplicate_lines': duplicate_lines,
            'skipped_lines': skipped_lines,
            'word_count': word_count,
            'error_metrics': error_metrics,
            'pause_metrics': pause_metrics,
            'repeated_words_length': len(repeated_words),
            'original_vs_audio': original_vs_audio,
            'accuracy': accuracy,
            'audio_duration': audio_duration,
            'transcription_confidence': 76,
            # Add other metrics as needed
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
