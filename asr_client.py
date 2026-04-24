#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3-ASR REST API Client

Example client for calling the ASR server from other machines.

Usage:
    python asr_client.py --url http://server-ip:5000 --audio path/to/audio.wav
"""

import argparse
import base64
import json
import logging
import os
import sys
from typing import Optional, Dict, Any

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ASRClient:
    """Client for Qwen3-ASR REST API."""

    def __init__(self, base_url: str = "http://localhost:5000", timeout: int = 300):
        """
        Initialize the ASR client.

        Args:
            base_url: Base URL of the ASR server (e.g., http://localhost:5000)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        logger.info(f"ASR Client initialized with server: {self.base_url}")

    def health_check(self) -> Dict[str, Any]:
        """Check server health."""
        response = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_languages(self) -> Dict[str, Any]:
        """Get supported languages."""
        response = self.session.get(f"{self.base_url}/languages", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def transcribe_file(
        self,
        audio_path: str,
        language: Optional[str] = None,
        context: Optional[str] = None,
        return_time_stamps: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to the audio file
            language: Optional language (e.g., "Chinese", "English")
            context: Optional context string
            return_time_stamps: Whether to return timestamps

        Returns:
            Dictionary with 'language', 'text', and optionally 'time_stamps'
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        files = {'audio': open(audio_path, 'rb')}
        data = {}

        if language:
            data['language'] = language
        if context:
            data['context'] = context
        if return_time_stamps:
            data['return_time_stamps'] = 'true'

        try:
            logger.info(f"Transcribing file: {audio_path}")
            response = self.session.post(
                f"{self.base_url}/transcribe",
                files=files,
                data=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        finally:
            files['audio'].close()

    def transcribe_url(
        self,
        audio_url: str,
        language: Optional[str] = None,
        context: Optional[str] = None,
        return_time_stamps: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe audio from a URL.

        Args:
            audio_url: URL to the audio file
            language: Optional language
            context: Optional context string
            return_time_stamps: Whether to return timestamps

        Returns:
            Dictionary with 'language', 'text', and optionally 'time_stamps'
        """
        data = {'audio_url': audio_url}

        if language:
            data['language'] = language
        if context:
            data['context'] = context
        if return_time_stamps:
            data['return_time_stamps'] = 'true'

        logger.info(f"Transcribing URL: {audio_url}")
        response = self.session.post(
            f"{self.base_url}/transcribe",
            json=data,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def transcribe_base64(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        context: Optional[str] = None,
        return_time_stamps: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe audio from base64-encoded bytes.

        Args:
            audio_bytes: Raw audio bytes
            language: Optional language
            context: Optional context string
            return_time_stamps: Whether to return timestamps

        Returns:
            Dictionary with 'language', 'text', and optionally 'time_stamps'
        """
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        data = {'audio_base64': audio_base64}

        if language:
            data['language'] = language
        if context:
            data['context'] = context
        if return_time_stamps:
            data['return_time_stamps'] = 'true'

        logger.info("Transcribing base64 audio")
        response = self.session.post(
            f"{self.base_url}/transcribe",
            json=data,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def transcribe_base64_from_file(
        self,
        audio_path: str,
        language: Optional[str] = None,
        context: Optional[str] = None,
        return_time_stamps: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe audio from a file by reading and encoding as base64.

        Args:
            audio_path: Path to the audio file
            language: Optional language
            context: Optional context string
            return_time_stamps: Whether to return timestamps

        Returns:
            Dictionary with 'language', 'text', and optionally 'time_stamps'
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()

        return self.transcribe_base64(audio_bytes, language, context, return_time_stamps)


def format_result(result: Dict[str, Any], show_time_stamps: bool = False) -> str:
    """Format the transcription result for display."""
    output = []
    output.append("-" * 60)
    output.append(f"Language: {result.get('language', 'Unknown')}")
    output.append(f"Text: {result.get('text', '')}")
    if show_time_stamps and result.get('time_stamps'):
        output.append("\nTime Stamps:")
        for ts in result['time_stamps']:
            output.append(f"  [{ts['start_time']:.2f}s - {ts['end_time']:.2f}s] {ts['text']}")
    output.append("-" * 60)
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description='Qwen3-ASR REST API Client')
    parser.add_argument('--url', type=str, default='http://10.223.12.17:5000',
                        help='Server URL (default: http://localhost:5000)')
    parser.add_argument('--audio', type=str,
                        help='Path to audio file to transcribe')
    parser.add_argument('--audio-url', type=str,
                        help='URL of audio file to transcribe')
    parser.add_argument('--language', type=str,
                        help='Language (e.g., Chinese, English, Japanese)')
    parser.add_argument('--context', type=str,
                        help='Context string')
    parser.add_argument('--time-stamps', action='store_true',
                        help='Return timestamps')
    parser.add_argument('--health-check', action='store_true',
                        help='Check server health')
    parser.add_argument('--list-languages', action='store_true',
                        help='List supported languages')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Request timeout in seconds (default: 300)')
    parser.add_argument('--output', type=str,
                        help='Save result to JSON file')

    args = parser.parse_args()

    client = ASRClient(base_url=args.url, timeout=args.timeout)

    try:
        # Health check
        if args.health_check:
            result = client.health_check()
            logger.info(f"Server status: {result['status']}")
            logger.info(f"Model loaded: {result['model_loaded']}")
            return

        # List languages
        if args.list_languages:
            result = client.get_languages()
            logger.info("Supported languages:")
            for lang in result.get('languages', []):
                logger.info(f"  - {lang}")
            return

        # Transcribe
        if args.audio:
            result = client.transcribe_file(
                audio_path=args.audio,
                language=args.language,
                context=args.context,
                return_time_stamps=args.time_stamps
            )
        elif args.audio_url:
            result = client.transcribe_url(
                audio_url=args.audio_url,
                language=args.language,
                context=args.context,
                return_time_stamps=args.time_stamps
            )
        else:
            parser.print_help()
            sys.exit(1)

        # Display result
        print(format_result(result, show_time_stamps=args.time_stamps))

        # Save to file if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Result saved to: {args.output}")

    except requests.exceptions.ConnectionError:
        logger.error(f"Failed to connect to server at {args.url}")
        logger.error("Please ensure the server is running and the URL is correct.")
        sys.exit(1)
    except requests.exceptions.Timeout:
        logger.error("Request timed out. The audio may be too long or the server is busy.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
