"""
VLM Client implementations for scene state extraction.

Supports: OpenAI GPT-4o, Anthropic Claude, Local models via Ollama
"""

import base64
import json
import os
import re
from abc import ABC, abstractmethod


def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 for API calls."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


SCENE_STATE_PROMPT = """Analyze this robot manipulation scene and describe the current state.

Focus ONLY on:
1. Objects on the table (name, location, contents if container)
2. Robot gripper state (open/closed/holding something)
3. Gripper position relative to objects

Respond in this exact JSON format:
{
    "objects": [
        {"name": "object name", "location": "where on table", "state": "empty/contains X/null"}
    ],
    "gripper_state": "open" | "closed" | "holding <object>",
    "gripper_position": "description of where gripper is"
}

Be precise. Only describe what you see. Return ONLY valid JSON."""


class VLMClient(ABC):
    @abstractmethod
    def get_scene_state(self, image_path: str) -> dict:
        """Query VLM for scene state, return parsed JSON."""
        pass


class OpenAIClient(VLMClient):
    """OpenAI GPT-4o client."""

    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI
        self.client = OpenAI()  # Uses OPENAI_API_KEY env var
        self.model = model

    def get_scene_state(self, image_path: str) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SCENE_STATE_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_image_base64(image_path)}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        return self._parse_json(response.choices[0].message.content)

    def _parse_json(self, text: str) -> dict:
        # Try to extract JSON from response
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON block in text
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                return json.loads(match.group())
            raise ValueError(f"Could not parse JSON from response: {text}")


class AnthropicClient(VLMClient):
    """Anthropic Claude client."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self.client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
        self.model = model

    def get_scene_state(self, image_path: str) -> dict:
        # Determine media type
        ext = image_path.lower().split('.')[-1]
        media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": encode_image_base64(image_path)
                            }
                        },
                        {
                            "type": "text",
                            "text": SCENE_STATE_PROMPT
                        }
                    ]
                }
            ]
        )
        return self._parse_json(response.content[0].text)

    def _parse_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                return json.loads(match.group())
            raise ValueError(f"Could not parse JSON from response: {text}")


class OllamaClient(VLMClient):
    """Local Ollama client for models like LLaVA."""

    def __init__(self, model: str = "llava:13b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host

    def get_scene_state(self, image_path: str) -> dict:
        import requests

        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": SCENE_STATE_PROMPT,
                "images": [encode_image_base64(image_path)],
                "stream": False
            }
        )
        response.raise_for_status()
        return self._parse_json(response.json()["response"])

    def _parse_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                return json.loads(match.group())
            raise ValueError(f"Could not parse JSON from response: {text}")


def get_client(provider: str = "openai", **kwargs) -> VLMClient:
    """Factory function to get VLM client."""
    providers = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "ollama": OllamaClient
    }
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(providers.keys())}")
    return providers[provider](**kwargs)
