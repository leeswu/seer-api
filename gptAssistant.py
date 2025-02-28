from openai import OpenAI
import base64
import pymupdf
from os import path


class GPTAssistant:
    def __init__(self, api_key, transcript):
        self.client = OpenAI(api_key=api_key)
        self.transcript = transcript
