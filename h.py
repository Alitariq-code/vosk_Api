from flask import Flask, request, jsonify

import requests
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from difflib import SequenceMatcher
from fun import analyze_audio, compare_lines, count_duplicate_lines, count_skipped_lines, count_words,calculate_word_count_ratio,get_wav_duration
app = Flask(__name__)

recognizer = sr.Recognizer()

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

def transcribe_audio_with_ffmpeg(audio_file_url, recognizer, language="nl-NL"):
    temp_file_path = "file.wav"  # Set the desired filename

    try:
        # Download the audio file directly from the URL
        response = requests.get(audio_file_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download audio from the provided URL. Status code: {response.status_code}")

        # Save audio data to a temporary file with the desired filename
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(response.content)

        # Provide the path to ffmpeg and ffprobe
        AudioSegment.converter = "/usr/bin/ffmpeg"
        AudioSegment.ffmpeg = "/usr/bin/ffmpeg"
        AudioSegment.ffprobe = "/usr/bin/ffprobe"

        # Convert audio to WAV format using pydub
        try:
            audio = AudioSegment.from_file(temp_file_path)
            audio.export('temp.wav', format="wav")
        except CouldntDecodeError:
            raise Exception("Failed to decode audio file. Check if the file is in a supported format.")

        # Use the recognizer directly without using pydub
        with sr.AudioFile('temp.wav') as source:
            recognizer.adjust_for_ambient_noise(source)

            audio = recognizer.record(source)
            response = recognizer.recognize_google(audio, language=language, show_all=True)

            if 'alternative' in response:
                # Extract the first alternative transcription
                alternative = response['alternative'][0]
                text = alternative.get('transcript', '')
                confidence = alternative.get('confidence', None)

                # Insert a line break when there's a pause in speech
                pauses = recognizer.pause_threshold
                text_with_line_breaks = ""
                for i, phrase in enumerate(text.split('\n')):
                    if i > 0:
                        text_with_line_breaks += '\n'

                    text_with_line_breaks += phrase

                    # Check if there is a pause between phrases
                    if i < len(text.split('\n')) - 1:
                        duration = recognizer.get_duration(audio)
                        if duration > pauses:
                            text_with_line_breaks += '\n'

                return text_with_line_breaks, confidence

            else:
                return "No transcription found in the response", None

    except sr.UnknownValueError:
        return "Could not understand audio", None

    except sr.RequestError as e:
        return f"Could not request results; {str(e)}", None

    finally:
        # No need to remove the temporary file since it's now "file.wav"
        pass

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        # Get data from the request
        data = request.get_json()
        audio_url = data.get('audio_url')
        original_text = data.get('original_text')

        # Call the functions
        transcribed_text, confidence = transcribe_audio_with_ffmpeg(audio_url, recognizer, language="nl-NL")

        # Additional analysis
        analysis_result = analyze_audio('temp.wav')
        deleted_words, inserted_words, substituted_words, repeated_words = compare_lines(original_text, transcribed_text)
        duplicate_lines = count_duplicate_lines(transcribed_text)
        skipped_lines = count_skipped_lines(transcribed_text)
        word_count = count_words(transcribed_text)

        # Calculate error metrics
        error_metrics = calculate_error_metrics(original_text, transcribed_text)

        # Calculate pause metrics
        pause_metrics = calculate_pause_metrics(transcribed_text)

        # Calculate accuracy, audio duration, and transcription confidence score
        accuracy = error_metrics.get('Words Correct per Minute', 0) / word_count * 100 if word_count != 0 else 0
        audio_duration = get_wav_duration('temp.wav')
        transcription_confidence = confidence if confidence is not None else 0

        original_vs_audio = calculate_word_count_ratio(transcribed_text, original_text)

        # Prepare JSON response with additional outcomes
        response = {
            'transcribed_text': transcribed_text,
            'confidence': confidence,  # Confidence score
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
            'transcription_confidence': transcription_confidence,
            # Add other metrics as needed
        }

        return jsonify(response)

    except Exception as e:
        # Handle exceptions and return an error response
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)