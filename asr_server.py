#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3-ASR REST API Server

A Flask-based REST API server for audio transcription using Qwen3-ASR.
Run this script to start the server, then access it via HTTP from other machines.

Usage:
    python asr_server.py

API Endpoints:
    POST /transcribe - Transcribe audio file or URL
    GET /health - Health check
    GET /languages - Get supported languages

Example request:
    curl -X POST http://localhost:5000/transcribe \
         -F "audio=@/path/to/audio.wav" \
         -F "language=Chinese"
"""

import argparse
import base64
import io
import json
import logging
import os
import tempfile
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
import soundfile as sf
import torch
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest

from qwen_asr import Qwen3ASRModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instance
asr_model: Optional[Qwen3ASRModel] = None

app = Flask(__name__)


def load_audio_from_file(file_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    try:
        audio, sr = sf.read(file_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # Convert to mono
        if sr != sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        return audio.astype(np.float32)
    except Exception as e:
        logger.error(f"Error loading audio from {file_path}: {e}")
        raise BadRequest(f"Failed to load audio file: {e}")


def load_audio_from_bytes(audio_bytes: bytes, sample_rate: int = 16000) -> np.ndarray:
    """Load audio from bytes."""
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr != sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        return audio.astype(np.float32)
    except Exception as e:
        logger.error(f"Error loading audio from bytes: {e}")
        raise BadRequest(f"Failed to load audio data: {e}")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_loaded': asr_model is not None,
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/languages', methods=['GET'])
def get_languages():
    """Get supported languages."""
    if asr_model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    return jsonify({
        'languages': asr_model.get_supported_languages()
    })


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Transcribe audio.

    Form parameters:
        audio: File upload (multipart/form-data)
        audio_url: URL to audio file (alternative to audio file)
        audio_base64: Base64-encoded audio data (alternative)
        language: Optional language (e.g., "Chinese", "English", "Japanese")
        context: Optional context string
        return_time_stamps: Boolean, whether to return timestamps (default: False)

    JSON parameters (alternative):
        audio_url: URL to audio file
        audio_base64: Base64-encoded audio data
        language: Optional language
        context: Optional context
        return_time_stamps: Boolean
    """
    if asr_model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    # Get audio input
    audio_input = None
    audio_source_type = None

    # Check for file upload
    if 'audio' in request.files:
        audio_file = request.files['audio']
        if audio_file.filename:
            # Save to temp file and pass path
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(audio_file.read())
                audio_input = tmp.name
                audio_source_type = 'file_upload'

    # Check for URL
    elif 'audio_url' in request.form or 'audio_url' in request.json:
        audio_url = request.form.get('audio_url') or request.json.get('audio_url')
        if audio_url:
            audio_input = audio_url
            audio_source_type = 'url'

    # Check for base64
    elif 'audio_base64' in request.form or 'audio_base64' in request.json:
        audio_base64 = request.form.get('audio_base64') or request.json.get('audio_base64')
        if audio_base64:
            try:
                # Remove data URL prefix if present
                if ',' in audio_base64:
                    audio_base64 = audio_base64.split(',', 1)[1]
                audio_bytes = base64.b64decode(audio_base64)
                # Save to temp file and pass path
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    tmp.write(audio_bytes)
                    audio_input = tmp.name
                    audio_source_type = 'base64'
            except Exception as e:
                return jsonify({'error': f'Failed to decode base64 audio: {e}'}), 400

    else:
        return jsonify({'error': 'No audio input provided. Use audio file, audio_url, or audio_base64.'}), 400

    # Get parameters
    language = request.form.get('language') or request.json.get('language')
    context = request.form.get('context') or request.json.get('context', '')
    return_time_stamps = request.form.get('return_time_stamps') or request.json.get('return_time_stamps', False)
    return_time_stamps = str(return_time_stamps).lower() in ('true', '1', 'yes')

    logger.info(f"Transcribing {audio_source_type}, language={language}, timestamps={return_time_stamps}")

    # Track temp file to clean up
    temp_file = audio_input if (audio_source_type in ('file_upload', 'base64')) else None

    try:
        # Run transcription
        results = asr_model.transcribe(
            audio=[audio_input],
            language=[language] if language else None,
            context=[context] if context else None,
            return_time_stamps=return_time_stamps
        )

        if not results:
            return jsonify({'error': 'No transcription result'}), 500

        result = results[0]

        # Format response
        response_data = {
            'language': result.language,
            'text': result.text,
            'time_stamps': None
        }

        # Add timestamps if requested
        if result.time_stamps is not None:
            timestamps = []
            for item in result.time_stamps.items:
                timestamps.append({
                    'text': item.text,
                    'start_time': item.start_time,
                    'end_time': item.end_time
                })
            response_data['time_stamps'] = timestamps

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        return jsonify({'error': f'Transcription failed: {str(e)}'}), 500
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_file}: {e}")


@app.errorhandler(Exception)
def handle_error(error):
    """Global error handler."""
    logger.error(f"Unhandled error: {error}", exc_info=True)
    return jsonify({'error': str(error)}), 500


def init_model(model_name: str, dtype: str, device: str,
               forced_aligner: str, max_inference_batch_size: int,
               max_new_tokens: int):
    """Initialize the ASR model."""
    global asr_model

    logger.info(f"Loading model: {model_name}")
    logger.info(f"Device: {device}, dtype: {dtype}")

    # Map dtype string to torch dtype
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    forced_aligner_kwargs = {
        'dtype': torch_dtype,
        'device_map': device
    }

    asr_model = Qwen3ASRModel.from_pretrained(
        pretrained_model_name_or_path=model_name,
        dtype=torch_dtype,
        device_map=device,
        forced_aligner=forced_aligner if forced_aligner else None,
        forced_aligner_kwargs=forced_aligner_kwargs if forced_aligner else None,
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=max_new_tokens,
    )

    logger.info("Model loaded successfully!")


def main():
    parser = argparse.ArgumentParser(description='Qwen3-ASR REST API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8080,
                        help='Port to bind to (default: 5000)')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-ASR-1.7B',
                        help='Model name or path (default: Qwen/Qwen3-ASR-1.7B)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (default: cuda:0)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float32', 'float16', 'bfloat16'],
                        help='Data type (default: bfloat16)')
    parser.add_argument('--forced-aligner', type=str, default='Qwen/Qwen3-ForcedAligner-0.6B',
                        help='Forced aligner model path (default: Qwen/Qwen3-ForcedAligner-0.6B)')
    parser.add_argument('--max-inference-batch-size', type=int, default=32,
                        help='Max inference batch size (default: 32)')
    parser.add_argument('--max-new-tokens', type=int, default=256,
                        help='Max new tokens to generate (default: 256)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')

    args = parser.parse_args()

    # Initialize model
    init_model(
        model_name=args.model,
        dtype=args.dtype,
        device=args.device,
        forced_aligner=args.forced_aligner,
        max_inference_batch_size=args.max_inference_batch_size,
        max_new_tokens=args.max_new_tokens
    )

    # Print server info
    logger.info("=" * 60)
    logger.info("Qwen3-ASR REST API Server")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Listening on: http://{args.host}:{args.port}")
    logger.info("=" * 60)
    logger.info("\nAvailable endpoints:")
    logger.info("  POST /transcribe  - Transcribe audio")
    logger.info("  GET  /health      - Health check")
    logger.info("  GET  /languages   - Get supported languages")
    logger.info("=" * 60)
    logger.info("\nExample usage:")
    logger.info(f"  curl -X POST http://{args.host}:{args.port}/transcribe \\")
    logger.info(f"       -F 'audio=@/path/to/audio.wav' \\")
    logger.info(f"       -F 'language=Chinese'")
    logger.info("=" * 60)

    # Run server
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
