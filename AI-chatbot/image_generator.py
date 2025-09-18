"""
Image generation module for StudyMate
Provides intelligent image generation with high-quality AI models
"""

import io
import base64
from PIL import Image, ImageDraw, ImageFont
import requests
import os
from typing import Optional, Tuple
import random
import time


class ImageGenerator:
    """Handle image generation with multiple high-quality API options"""

    def __init__(self):
        self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

    def generate_image(self, prompt: str) -> Tuple[bool, Optional[str], str]:
        """
        Generate an image from text prompt using high-quality APIs
        Returns: (success, base64_image, message)
        """
        try:
            return self._generate_with_pollinations(prompt)
        except Exception as e:
            print(f"Pollinations API failed: {e}")

        # Try Hugging Face API with API key
        if self.huggingface_api_key:
            try:
                return self._generate_with_huggingface(prompt)
            except Exception as e:
                print(f"Hugging Face API failed: {e}")

        # Try free Hugging Face inference (no API key required)
        try:
            return self._generate_with_free_huggingface(prompt)
        except Exception as e:
            print(f"Free Hugging Face failed: {e}")

        try:
            return self._generate_with_picsum(prompt)
        except Exception as e:
            print(f"Picsum failed: {e}")

        # Final fallback to visual drawings
        return self._generate_visual_image(prompt)

    def _generate_with_pollinations(self, prompt: str) -> Tuple[bool, Optional[str], str]:
        """Generate image using Pollinations API (completely free)"""
        enhanced_prompt = f"high quality, detailed, {prompt}, professional photography, 4k"

        import urllib.parse

        encoded_prompt = urllib.parse.quote(enhanced_prompt)

        api_url = (
            f"https://image.pollinations.ai/prompt/{encoded_prompt}"
            f"?width=128&height=128&model=flux&seed={random.randint(1, 1000000)}"
        )

        response = requests.get(api_url, timeout=30)

        if response.status_code == 200 and response.content:
            if response.headers.get("content-type", "").startswith("image/"):
                image_b64 = base64.b64encode(response.content).decode()
                return True, image_b64, f"Generated high-quality AI image: {prompt}"
            else:
                raise Exception("Response is not an image")
        else:
            raise Exception(f"API returned status {response.status_code}")

    def _generate_visual_image(self, prompt: str) -> Tuple[bool, str, str]:
        """Generate a meaningful visual image based on the prompt"""
        width, height = 128, 128  # Reduced size

        prompt_lower = prompt.lower()

        if any(word in prompt_lower for word in ["boy", "man", "person", "human", "people"]):
            return self._create_person_image(prompt, width, height)
        elif any(word in prompt_lower for word in ["nature", "tree", "forest", "landscape", "mountain"]):
            return self._create_nature_image(prompt, width, height)
        elif any(word in prompt_lower for word in ["animal", "cat", "dog", "bird", "lion", "tiger"]):
            return self._create_animal_image(prompt, width, height)
        elif any(word in prompt_lower for word in ["building", "house", "city", "architecture"]):
            return self._create_building_image(prompt, width, height)
        else:
            return self._create_abstract_image(prompt, width, height)

    def _generate_with_huggingface(self, prompt: str) -> Tuple[bool, Optional[str], str]:
        """Generate image using Hugging Face API with API key"""
        api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

        headers = {"Authorization": f"Bearer {self.huggingface_api_key}"}
        data = {"inputs": f"high quality, detailed, {prompt}"}

        response = requests.post(api_url, headers=headers, json=data, timeout=30)

        if response.status_code == 200:
            image_b64 = base64.b64encode(response.content).decode()
            return True, image_b64, f"Generated AI image: {prompt}"
        else:
            raise Exception(f"API returned status {response.status_code}")

    def _generate_with_free_huggingface(self, prompt: str) -> Tuple[bool, Optional[str], str]:
        """Generate image using free Hugging Face inference"""
        models = [
            "runwayml/stable-diffusion-v1-5",
            "CompVis/stable-diffusion-v1-4",
            "stabilityai/stable-diffusion-2-1",
        ]

        for model in models:
            try:
                api_url = f"https://api-inference.huggingface.co/models/{model}"
                data = {"inputs": f"high quality, detailed, {prompt}"}

                response = requests.post(api_url, json=data, timeout=30)

                if response.status_code == 200:
                    image_b64 = base64.b64encode(response.content).decode()
                    return True, image_b64, f"Generated AI image: {prompt}"
                elif response.status_code == 503:
                    # Model is loading, wait and try next
                    time.sleep(2)
                    continue
            except Exception:
                continue

        raise Exception("All Hugging Face models failed")

    def _generate_with_picsum(self, prompt: str) -> Tuple[bool, Optional[str], str]:
        """Generate image using Picsum for photographic content"""
        prompt_lower = prompt.lower()

        # Only use Picsum for nature, landscape, or general photo requests
        if any(word in prompt_lower for word in ["nature", "landscape", "photo", "scene", "view"]):
            api_url = f"https://picsum.photos/128/128?random={random.randint(1, 1000)}"

            response = requests.get(api_url, timeout=15)

            if response.status_code == 200:
                image_b64 = base64.b64encode(response.content).decode()
                return True, image_b64, f"Generated photographic image: {prompt}"

        raise Exception("Picsum not suitable for this prompt")

    def _create_person_image(self, prompt: str, width: int, height: int) -> Tuple[bool, str, str]:
        """Create a simple person representation"""
        image = Image.new("RGB", (width, height), color="#87CEEB")  # Sky blue background
        draw = ImageDraw.Draw(image)

        # Draw simple person figure
        center_x, center_y = width // 2, height // 2

        # Head
        head_radius = 15
        draw.ellipse(
            [center_x - head_radius, center_y - 40, center_x + head_radius, center_y - 10],
            fill="#FDBCB4",
        )

        # Body
        draw.rectangle([center_x - 10, center_y - 10, center_x + 10, center_y + 30], fill="#4169E1")

        # Arms
        draw.rectangle([center_x - 25, center_y - 5, center_x - 10, center_y + 5], fill="#FDBCB4")
        draw.rectangle([center_x + 10, center_y - 5, center_x + 25, center_y + 5], fill="#FDBCB4")

        # Legs
        draw.rectangle([center_x - 8, center_y + 30, center_x - 2, center_y + 55], fill="#000080")
        draw.rectangle([center_x + 2, center_y + 30, center_x + 8, center_y + 55], fill="#000080")

        self._add_title(draw, prompt, width, height)
        return True, self._image_to_base64(image), f"Created visual representation: {prompt}"

    def _create_nature_image(self, prompt: str, width: int, height: int) -> Tuple[bool, str, str]:
        """Create a nature scene"""
        image = Image.new("RGB", (width, height), color="#87CEEB")  # Sky
        draw = ImageDraw.Draw(image)

        # Ground
        draw.rectangle([0, height * 0.7, width, height], fill="#228B22")

        # Sun
        draw.ellipse([width * 0.75, height * 0.05, width * 0.9, height * 0.2], fill="#FFD700")

        # Trees
        for i in range(2):
            x = width * (0.25 + i * 0.4)
            draw.rectangle([x - 3, height * 0.55, x + 3, height * 0.7], fill="#8B4513")
            draw.ellipse([x - 15, height * 0.35, x + 15, height * 0.6], fill="#006400")

        self._add_title(draw, prompt, width, height)
        return True, self._image_to_base64(image), f"Created nature scene: {prompt}"

    def _create_animal_image(self, prompt: str, width: int, height: int) -> Tuple[bool, str, str]:
        """Create a simple animal representation"""
        image = Image.new("RGB", (width, height), color="#98FB98")  # Light green background
        draw = ImageDraw.Draw(image)

        center_x, center_y = width // 2, height // 2

        # Body
        draw.ellipse([center_x - 20, center_y - 5, center_x + 20, center_y + 15], fill="#FFA500")

        # Head
        draw.ellipse([center_x - 12, center_y - 25, center_x + 12, center_y], fill="#FFA500")

        # Ears
        draw.polygon(
            [center_x - 10, center_y - 25, center_x - 5, center_y - 35, center_x, center_y - 25],
            fill="#FFA500",
        )
        draw.polygon(
            [center_x, center_y - 25, center_x + 5, center_y - 35, center_x + 10, center_y - 25],
            fill="#FFA500",
        )

        # Eyes
        draw.ellipse([center_x - 7, center_y - 18, center_x - 3, center_y - 14], fill="black")
        draw.ellipse([center_x + 3, center_y - 18, center_x + 7, center_y - 14], fill="black")

        self._add_title(draw, prompt, width, height)
        return True, self._image_to_base64(image), f"Created animal illustration: {prompt}"

    def _create_building_image(self, prompt: str, width: int, height: int) -> Tuple[bool, str, str]:
        """Create a building/architecture scene"""
        image = Image.new("RGB", (width, height), color="#87CEEB")
        draw = ImageDraw.Draw(image)

        # Ground
        draw.rectangle([0, height * 0.8, width, height], fill="#696969")

        # Buildings
        for i in range(2):
            x = 30 + i * 60
            h = 60 + i * 20
            w = 30 + i * 10

            draw.rectangle([x, height - h, x + w, height * 0.8], fill="#708090")

        self._add_title(draw, prompt, width, height)
        return True, self._image_to_base64(image), f"Created building scene: {prompt}"

    def _create_abstract_image(self, prompt: str, width: int, height: int) -> Tuple[bool, str, str]:
        """Create an abstract representation"""
        image = Image.new("RGB", (width, height), color="#2C3E50")
        draw = ImageDraw.Draw(image)

        colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]

        for i in range(6):
            color = random.choice(colors)
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)

            if i % 2 == 0:
                draw.ellipse([x1, y1, x2, y2], fill=color)
            else:
                draw.rectangle([x1, y1, x2, y2], fill=color)

        self._add_title(draw, prompt, width, height)
        return True, self._image_to_base64(image), f"Created abstract art: {prompt}"

    def _add_title(self, draw, prompt: str, width: int, height: int):
        """Add title to the image"""
        try:
            font = ImageFont.truetype("arial.ttf", 8)  # Smaller font
        except:
            font = ImageFont.load_default()

        draw.rectangle([2, height - 15, width - 2, height - 2], fill="black")
        text = prompt[:15] + "..." if len(prompt) > 15 else prompt
        draw.text((5, height - 12), text, fill="white", font=font)

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
