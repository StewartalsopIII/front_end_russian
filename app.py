import os
import asyncio
import json
import base64
import logging
import sys
import warnings
import threading
from dotenv import load_dotenv
from flask import Flask, render_template
from flask_socketio import SocketIO
from openai import AsyncOpenAI
from elevenlabs import Voice, VoiceSettings
from elevenlabs.client import ElevenLabs

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Field .* has conflict with protected namespace")

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # You can change this to your preferred voice ID

# Create ElevenLabs client
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    logger.info("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected")

async def text_chunker(chunks):
    buffer = ""
    async for text in chunks:
        if buffer.endswith((".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")):
            yield buffer + " "
            buffer = text
        elif text.startswith((".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")):
            yield buffer + text[0] + " "
            buffer = text[1:]
        else:
            buffer += text
    if buffer:
        yield buffer + " "

async def text_to_speech_streaming(text):
    logger.info("Starting ElevenLabs text-to-speech conversion")
    try:
        audio = elevenlabs_client.generate(
            text=text,
            voice=Voice(
                voice_id=VOICE_ID,
                settings=VoiceSettings(stability=0.5, similarity_boost=0.75)
            ),
            model="eleven_multilingual_v2",
            stream=True
        )
        
        for chunk in audio:
            yield chunk
        
        logger.info("Finished ElevenLabs text-to-speech conversion")
    except Exception as e:
        logger.error(f"Error in text_to_speech_streaming: {str(e)}", exc_info=True)
        raise

async def translate_and_generate_speech(text, target_language):
    logger.info(f"Starting OpenAI translation for: '{text}' to {target_language}")
    try:
        response = await openai_client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'system', 'content': f'You are a translator. Translate the following text to {target_language}.'},
                {'role': 'user', 'content': text}
            ],
            temperature=0.7,
            stream=True
        )
        logger.info("OpenAI translation stream created successfully")

        translated_text = ""
        async for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                translated_text += delta.content
                logger.debug(f"Received translation chunk: {delta.content}")
        
        logger.info(f"Full translation: '{translated_text}'")

        logger.info("Starting ElevenLabs text-to-speech conversion")
        async for audio_chunk in text_to_speech_streaming(translated_text):
            yield audio_chunk

    except Exception as e:
        logger.error(f"Error in translate_and_generate_speech: {str(e)}", exc_info=True)
        raise

@socketio.on('translate_and_speak')
def handle_translate_and_speak(data):
    logger.info(f"Received translate_and_speak request: {data}")
    text = data['text']
    target_language = data['target_language']
    
    def run_async_task():
        async def async_task():
            try:
                logger.info(f"Starting translation for text: '{text}' to language: {target_language}")
                audio_chunks = []
                async for audio_chunk in translate_and_generate_speech(text, target_language):
                    audio_chunks.append(audio_chunk)
                
                # Combine all audio chunks into a single audio data
                full_audio = b''.join(audio_chunks)
                logger.debug(f"Total audio size: {len(full_audio)} bytes")
                
                # Emit the full audio data as a single message
                socketio.emit('audio_data', {'audio': base64.b64encode(full_audio).decode('utf-8')})
                logger.info("Finished emitting audio data")
            except Exception as e:
                logger.error(f"Error in translate_and_speak: {str(e)}", exc_info=True)
                socketio.emit('error', {'message': str(e)})

        asyncio.run(async_task())

    # Run the async task in a separate thread
    thread = threading.Thread(target=run_async_task)
    thread.start()

if __name__ == '__main__':
    logger.info("Starting Flask-SocketIO server")
    socketio.run(app, debug=True)