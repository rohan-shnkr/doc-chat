#Step 1: Get the audio file (input from the user)
#Step 2: Transcribe and translate the audio file to english (speech_to_text.py)
#Step 3: Get query answered from the document (rag_agent.py)
#Step 4: Translate the answer to the language of the audio file (translation.py)
#Step 5: Text to speech (text_to_speech.py)
#Step 6: Play the speech (output here)

import rag_agent
import speech_to_text
import translation
import text_to_speech
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import os
from pygame import mixer
import pygame

LANGUAGE_MAPPING_DICT={
        'af': 'afr_Latn',  # Afrikaans
        'am': 'amh_Ethi',  # Amharic
        'ar': 'arb_Arab',  # Arabic
        'as': 'asm_Beng',  # Assamese
        'az': 'azj_Latn',  # Azerbaijani
        'ba': 'bak_Cyrl',  # Bashkir
        'be': 'bel_Cyrl',  # Belarusian
        'bg': 'bul_Cyrl',  # Bulgarian
        'bn': 'ben_Beng',  # Bengali
        'bo': 'bod_Tibt',  # Tibetan
        'bs': 'bos_Latn',  # Bosnian
        'ca': 'cat_Latn',  # Catalan
        'cs': 'ces_Latn',  # Czech
        'cy': 'cym_Latn',  # Welsh
        'da': 'dan_Latn',  # Danish
        'de': 'deu_Latn',  # German
        'el': 'ell_Grek',  # Greek
        'en': 'eng_Latn',  # English
        'es': 'spa_Latn',  # Spanish
        'et': 'est_Latn',  # Estonian
        'eu': 'eus_Latn',  # Basque
        'fa': 'pes_Arab',  # Persian
        'fi': 'fin_Latn',  # Finnish
        'fo': 'fao_Latn',  # Faroese
        'fr': 'fra_Latn',  # French
        'gl': 'glg_Latn',  # Galician
        'gu': 'guj_Gujr',  # Gujarati
        'ha': 'hau_Latn',  # Hausa
        'he': 'heb_Hebr',  # Hebrew
        'hi': 'hin_Deva',  # Hindi
        'hr': 'hrv_Latn',  # Croatian
        'ht': 'hat_Latn',  # Haitian Creole
        'hu': 'hun_Latn',  # Hungarian
        'hy': 'hye_Armn',  # Armenian
        'id': 'ind_Latn',  # Indonesian
        'is': 'isl_Latn',  # Icelandic
        'it': 'ita_Latn',  # Italian
        'ja': 'jpn_Jpan',  # Japanese
        'jw': 'jav_Latn',  # Javanese
        'ka': 'kat_Geor',  # Georgian
        'kk': 'kaz_Cyrl',  # Kazakh
        'km': 'khm_Khmr',  # Khmer
        'kn': 'kan_Knda',  # Kannada
        'ko': 'kor_Hang',  # Korean
        'lo': 'lao_Laoo',  # Lao
        'lt': 'lit_Latn',  # Lithuanian
        'lv': 'lvs_Latn',  # Latvian
        'mk': 'mkd_Cyrl',  # Macedonian
        'ml': 'mal_Mlym',  # Malayalam
        'mn': 'khk_Cyrl',  # Mongolian
        'mr': 'mar_Deva',  # Marathi
        'ms': 'zsm_Latn',  # Malay
        'mt': 'mlt_Latn',  # Maltese
        'my': 'mya_Mymr',  # Burmese
        'ne': 'npi_Deva',  # Nepali
        'nl': 'nld_Latn',  # Dutch
        'nn': 'nno_Latn',  # Norwegian Nynorsk
        'no': 'nob_Latn',  # Norwegian Bokmål
        'oc': 'oci_Latn',  # Occitan
        'pa': 'pan_Guru',  # Punjabi
        'pl': 'pol_Latn',  # Polish
        'ps': 'pbt_Arab',  # Pashto
        'pt': 'por_Latn',  # Portuguese
        'ro': 'ron_Latn',  # Romanian
        'ru': 'rus_Cyrl',  # Russian
        'sa': 'san_Deva',  # Sanskrit
        'sd': 'snd_Arab',  # Sindhi
        'si': 'sin_Sinh',  # Sinhala
        'sk': 'slk_Latn',  # Slovak
        'sl': 'slv_Latn',  # Slovenian
        'sn': 'sna_Latn',  # Shona
        'so': 'som_Latn',  # Somali
        'sq': 'als_Latn',  # Albanian
        'sr': 'srp_Cyrl',  # Serbian
        'su': 'sun_Latn',  # Sundanese
        'sv': 'swe_Latn',  # Swedish
        'sw': 'swh_Latn',  # Swahili
        'ta': 'tam_Taml',  # Tamil
        'te': 'tel_Telu',  # Telugu
        'tg': 'tgk_Cyrl',  # Tajik
        'th': 'tha_Thai',  # Thai
        'tk': 'tuk_Latn',  # Turkmen
        'tl': 'tgl_Latn',  # Tagalog
        'tr': 'tur_Latn',  # Turkish
        'tt': 'tat_Cyrl',  # Tatar
        'uk': 'ukr_Cyrl',  # Ukrainian
        'ur': 'urd_Arab',  # Urdu
        'uz': 'uzn_Latn',  # Uzbek
        'vi': 'vie_Latn',  # Vietnamese
        'yi': 'ydd_Hebr',  # Yiddish
        'yo': 'yor_Latn',  # Yoruba
        'zh': 'zho_Hans',  # Chinese (Simplified)
        'yue': 'yue_Hant', # Chinese (Cantonese)
    }

def record_audio(silence_threshold=0.005, silence_duration=5, sample_rate=44100, recording_file="user_recording.wav"):
    """
    Record audio from the microphone until silence is detected
    silence_threshold: amplitude threshold below which audio is considered silence
    silence_duration: duration of silence (in seconds) before stopping
    sample_rate: sample rate in Hz
    """
    # Create user_input directory if it doesn't exist
    if not os.path.exists('user_input'):
        os.makedirs('user_input')
    
    print("Recording... (speak now, will stop after 5 seconds of silence)")
    
    # Initialize variables for silence detection
    silence_samples = int(silence_duration * sample_rate)
    buffer = []
    silent_frames = 0
    
    def callback(indata, frames, time, status):
        nonlocal silent_frames
        if status:
            print(status)
        buffer.extend(indata[:, 0])  # Only take first channel
        # Check if the latest frame is silent
        if np.abs(indata).mean() < silence_threshold:
            silent_frames += len(indata)
        else:
            silent_frames = 0
    
    # Start recording
    with sd.InputStream(samplerate=sample_rate, 
                       channels=1, 
                       callback=callback):
        while silent_frames < silence_samples:
            sd.sleep(100)  # Sleep for 100ms
    
    print("Recording finished!")
    
    # Convert buffer to numpy array
    recording = np.array(buffer)
    
    # Save as WAV
    wav_path = f'user_input/{recording_file}'
    wav.write(wav_path, sample_rate, recording)
    
    return wav_path

def process_query(audio_file):
    extracted_text_dict = speech_to_text.transcribe_audio(audio_file_path=audio_file)
    # print(extracted_text_dict)
    extracted_text = extracted_text_dict['text']
    extracted_src_lang = extracted_text_dict['inferred_languages'][0]
    
    return extracted_text, extracted_src_lang

def process_rag_output(text, tgt_lang):
    #Translate to tgt_lang
    print("Language sent to translation:", LANGUAGE_MAPPING_DICT[tgt_lang])
    translated_dict = translation.translate_text(text, src_lang='eng_Latn', tgt_lang=LANGUAGE_MAPPING_DICT[tgt_lang])
    print(translated_dict[0]['translation_text'])
    translated_text = translated_dict[0]['translation_text']

    return translated_text

def voice_output(text, voice_id="pNInz6obpgDQGcFmaJgB"):
    pygame.init()
    mixer.init()
    
    audio_stream = text_to_speech.text_to_speech_stream(text, voice_id)
    # Load and play the audio using pygame
    mixer.music.load(audio_stream)
    mixer.music.play()
    
    # Wait for the audio to finish playing
    while mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def run_voice_chat(filename):
    rag_chain = rag_agent.setup_pdf_rag(filename=filename)

    try:
        while True:
            audio_file = record_audio()
            print(f"Audio saved to: {audio_file}")
            extracted_text, extracted_lang = process_query(audio_file=audio_file)
            print(extracted_text, extracted_lang)
            rag_response = rag_agent.rag_agent_response({"input": f"{extracted_text}"}, rag_chain)
            rag_response_text = rag_response["answer"]
            translated_text = process_rag_output(rag_response_text, extracted_lang)
            voice_output(translated_text)
            print("Bot has finished answer. Your turn")
    except KeyboardInterrupt:
        print('interrupted!')

    print("Chat ended")
    return

if __name__ == "__main__":
    run_voice_chat("./test-file/sample-pdf.pdf")
