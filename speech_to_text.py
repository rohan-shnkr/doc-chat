import fal_client

def on_queue_update(update):
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
           print(log["message"])

def transcribe_audio(audio_file_path):
    url = fal_client.upload_file(audio_file_path)
    result = fal_client.subscribe(
        "fal-ai/whisper",
        arguments={
            "audio_url": url,
            "task": "translate",
            "chunk_level": "segment",
            "version": "3",
            "batch_size": 64,
            "num_speakers": None
        },
        with_logs=True,
        on_queue_update=on_queue_update,
    )
    return result

if __name__ == "__main__":
    result = transcribe_audio("./user_input/user_recording.wav")
    print(result)
