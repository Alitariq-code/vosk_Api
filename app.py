from vosk import Model, KaldiRecognizer
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydub import AudioSegment
import time
from fun import (
    analyze_audio,
    compare_lines,
    count_duplicate_lines,
    count_skipped_lines,
    count_words,
    calculate_word_count_ratio,
    get_wav_duration,
)
from test import transcribe_audio_file
import requests

app = Flask(__name__)
CORS(app, origins="*")

def calculate_words_per_minute(words_correct, transcribed_text):
    audio_duration = get_wav_duration('temp.wav')
    
    # Check if audio_duration is greater than zero before division
    if audio_duration > 0:
        words_per_minute = (words_correct / audio_duration) * 60
    else:
        # Set a default value or handle it as per your requirements
        words_per_minute = None

    return words_per_minute


def calculate_error_metrics(original_text, transcribed_text, delete, insert, sub, manual_text):
    words_original = original_text.split()
    words_transcribed = transcribed_text.split()

    if manual_text:
        words_manual_text = manual_text.split()
        wt_maual = len(words_manual_text)
    else:
        words_manual_text = []
        wt_maual = 0

    wr = len(words_transcribed)
    wt_original = len(words_original)
    wc = wt_original - (delete + insert + sub)

    acc = wc * 100 / wt_original
    wc = max(0, wc)
    acc = max(0, acc)
    oriVsTran = 100 * min(wt_original / wr, wr / wt_original)
    manualVsTrans = 100 * min(wt_maual / wr, wr / wt_maual) if wt_maual > 0 else 0
    manualVsorginal = 100 * min(wt_maual / wt_original, wt_original / wt_maual) if wt_maual > 0 else 0

    error_metrics = {
        'WR': wr,
        'WC': wc,
        'Words Correct per Minute': calculate_words_per_minute(wc, transcribed_text),
        'Acc': acc,
        'oriVsTran': oriVsTran,
        'manualVsTrans': manualVsTrans,
        'manualVsorginal': manualVsorginal
    }

    return error_metrics

def calculate_word_count_ratio(transcribed_text, original_text, max_ratio=100):
    word_count_org = count_words(original_text)
    word_count_transcribed = count_words(transcribed_text)

    ratio = min(100*word_count_org / word_count_transcribed , word_count_transcribed* 100/word_count_org)
    return ratio

def calculate_pause_metrics(transcribed_text):
    # Your logic for pause metrics calculation goes here
    # Placeholder values are used, replace them with actual calculations
    pauses_1_3_seconds = 5
    hesitations_3_seconds = 2

    pause_metrics = {
        'Pauses (1-3 seconds)': pauses_1_3_seconds,
        'Hesitations (3+ seconds)': hesitations_3_seconds,
        # Add other pause metrics as needed
    }

    return pause_metrics

def format_word_list(word_list):
    return [{'ID': entry['ID'], 'Word': entry['Word']} for entry in word_list]
bufferData=[]
@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        data = request.get_json()
        audio_url = data.get('audio_url')
        original_text = data.get('original_text')
       
        manual_text= data.get('manual_text')
        transcrib = ''
        id = data.get('id')
        # Download the audio file from the provided URL
        if id and audio_url:
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
            transcribed_text = transcribe_audio_file("temp.wav")
            newData = {}
            newData["id"] = id
            newData["Text"] = transcribed_text
            bufferData.append(newData)
            print(bufferData)
            response = {
            'staus': 'done with audio at vosk Api'
        }
            return jsonify(response)

        else:
            print("okok")
            print("Searching for ID:", id)

            start_time = time.time()
            timeout_duration = 90  # 90 seconds = 1.5 minutes
            id_found = False

            while not id_found and time.time() - start_time < timeout_duration:
                elapsed_time = time.time() - start_time
                print(f"Elapsed time: {elapsed_time:.2f} seconds")

                for item in bufferData:
                    if 'id' in item and item['id'] == id:
                        transcrib = item['Text']
                        print("Data of this:", transcrib)
                        id_found = True  # Set the flag to True when ID is found
                        break      
        # print(transcribed_text)
            substituted_words,delete_words,insert_words,merged= compare_lines(original_text, transcrib)
            
            # error_metrics = calculate_error_metrics(original_text, transcribed_text)

            # Calculate pause metrics
            pause_metrics = calculate_pause_metrics(transcrib)

            # Convert DataFrames to lists of dictionaries
            delete, insert, sub = len(delete_words), len(insert_words), len(substituted_words)
            print(delete,insert,sub)
            error_metrics = calculate_error_metrics(original_text, transcrib, delete, insert, sub, manual_text)

            formatted_deleted_words = format_word_list(delete_words)
            formatted_inserted_words = format_word_list(insert_words)
            formatted_substituted_words = format_word_list(substituted_words)
            merged_format = format_word_list(merged)


        
            accuracy=error_metrics['Acc']

            original_vs_audio = calculate_word_count_ratio(transcrib, original_text)

            transcription_confidence = 76  

            audio_duration = get_wav_duration('temp.wav')

            # original_vs_audio = calculate_word_count_ratio(transcribed_text, original_text)
            analysis_result = analyze_audio('temp.wav')
            # Prepare JSON response with additional outcomes
            response = {
        'transcribed_text': transcrib,
        'analysis_result': analysis_result,
        'deleted_words': formatted_deleted_words,
        'inserted_words': formatted_inserted_words,
        'merged':merged,
        'substituted_words': formatted_substituted_words,
        'duplicate_lines': count_duplicate_lines(transcrib),
        'skipped_lines': count_skipped_lines(transcrib),
        # 'word_count': word_count,
        'error_metrics': error_metrics,
        'pause_metrics': pause_metrics,
        'original_vs_audio': error_metrics['oriVsTran'],
        'manualVsTrans':error_metrics['manualVsTrans'],
        'manualVsorginal':error_metrics['manualVsorginal'],
        'accuracy': accuracy,
        'audio_duration': audio_duration,
        'transcription_confidence': transcription_confidence,
        # Add other metrics as needed
    }
            return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})
# @app.route('/process_audio', methods=['POST'])
# def process_audio():
#     try:
#         data = request.get_json()
#         audio_url = data.get('audio_url')
#         original_text = data.get('original_text')

#         # Download the audio file from the provided URL
#         response = requests.get(audio_url)
#         if response.status_code != 200:
#             raise Exception(f"Failed to download audio from the provided URL. Status code: {response.status_code}")

#         # Save audio data to a temporary file
#         audio_file_path = "temp_original.wav"
#         with open(audio_file_path, "wb") as temp_file:
#             temp_file.write(response.content)

#         # Convert audio to mono and set sample width to 2 bytes
#         audio = AudioSegment.from_file(audio_file_path)
#         audio = audio.set_channels(1).set_sample_width(2)
#         audio.export("temp.wav", format="wav")

#         # Transcribe audio using Vosk
#         transcribed_text = transcribe_audio_file("temp.wav")
#         # print(transcribed_text)
#         substituted_words,delete_words,insert_words= compare_lines(original_text, transcribed_text)
        
#         # error_metrics = calculate_error_metrics(original_text, transcribed_text)

#         # Calculate pause metrics
#         pause_metrics = calculate_pause_metrics(transcribed_text)

#         # Convert DataFrames to lists of dictionaries
#         delete, insert, sub = len(delete_words), len(insert_words), len(substituted_words)
#         print(delete,insert,sub)
#         error_metrics = calculate_error_metrics(original_text, transcribed_text, delete=delete, insert=insert, sub=sub)

#         formatted_deleted_words = format_word_list(delete_words)
#         formatted_inserted_words = format_word_list(insert_words)
#         formatted_substituted_words = format_word_list(substituted_words)


       
#         accuracy=error_metrics['Acc']

#         original_vs_audio = calculate_word_count_ratio(transcribed_text, original_text)

#         transcription_confidence = 76  

#         audio_duration = get_wav_duration('temp.wav')

#         # original_vs_audio = calculate_word_count_ratio(transcribed_text, original_text)
#         analysis_result = analyze_audio('temp.wav')
#         # Prepare JSON response with additional outcomes
#         response = {
#     'transcribed_text': transcribed_text,
#     'analysis_result': analysis_result,
#     'deleted_words': formatted_deleted_words,
#     'inserted_words': formatted_inserted_words,
#     'substituted_words': formatted_substituted_words,
#     'duplicate_lines': count_duplicate_lines(transcribed_text),
#     'skipped_lines': count_skipped_lines(transcribed_text),
#     # 'word_count': word_count,
#     'error_metrics': error_metrics,
#     'pause_metrics': pause_metrics,
#     'original_vs_audio': original_vs_audio,
#     'accuracy': accuracy,
#     'audio_duration': audio_duration,
#     'transcription_confidence': transcription_confidence,
#     # Add other metrics as needed
# }
#         return jsonify(response)

#     except Exception as e:
#         return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
