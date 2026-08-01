import hashlib
import os
import time
from pathlib import Path
from google import genai
from google.genai import types
from typing import Optional, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

# Regenerating a dish that was already drawn is pure waste, and repeat visitors
# to the demo hit the same sample menu, so cache on disk by prompt hash.
CACHE_DIR = Path('.menu_vision_cache')

# Nano Banana 2. Imagen 4 Fast was 10 RPM / 70 RPD and is no longer served to
# newly issued keys; this model allows 100 RPM / 1000 RPD, so the daily ceiling
# stops being the thing that shapes the whole app.
IMAGE_MODEL = 'gemini-3.1-flash-image'

MAX_ATTEMPTS = 3
RETRY_SLEEP_SECONDS = 2


def cache_key(prompt: str, style: str = "") -> str:
    """Stable id for a prompt/style pair. Style is part of the key because the
    same dish rendered in a different restaurant style is a different image."""
    return hashlib.sha256(f'{prompt}||{style}'.encode('utf-8')).hexdigest()


def cached_image(key: str) -> Optional[bytes]:
    path = Path(CACHE_DIR) / f'{key}.png'
    return path.read_bytes() if path.exists() else None


def store_image(key: str, data: bytes) -> None:
    directory = Path(CACHE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'{key}.png').write_bytes(data)


def _get_client():
    """Get or create a Gemini API client."""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")
    return genai.Client(api_key=api_key)


def _call_image_model(client, prompt: str) -> Optional[bytes]:
    """Generate one image and return its raw bytes, or None if the model
    returned no image.

    Isolated so tests can stub the network call and the retry wrapper has a
    single, well-defined boundary to catch failures around.

    Uses generate_content rather than generate_images: the Imagen models are no
    longer served to newly issued API keys (they return 404 'no longer available
    to new users') and the generate_images method itself is deprecated.
    """
    response = client.models.generate_content(model=IMAGE_MODEL, contents=prompt)
    for candidate in response.candidates or []:
        for part in candidate.content.parts or []:
            blob = getattr(part, 'inline_data', None)
            if blob and blob.data:
                return blob.data
    return None


def generate_image(menu_item: Dict[str, Any], restaurant_style: str = "") -> Optional[Dict[str, Any]]:
    """
    Generates an image for a menu item.

    Args:
        menu_item: Dictionary containing menu item data with 'name' and 'prompt'
        restaurant_style: Optional style string to enforce visual consistency across all dishes.

    Returns:
        Dictionary with menu item data plus 'image_bytes' field, or None if generation fails
    """
    try:
        # Extract the prompt from the menu item
        prompt = str(menu_item.get('prompt', ''))
        if not prompt:
            print(f"No prompt found for menu item: {menu_item.get('name', 'Unknown')}")
            return None

        # Append restaurant style for visual consistency across all generated images
        if restaurant_style:
            prompt = f"{prompt} Shot in the style of: {restaurant_style}."

        key = cache_key(prompt, restaurant_style)
        cached = cached_image(key)
        if cached is not None:
            print(f"💾 Using cached image for: {menu_item.get('name', 'Unknown')}")
            result = menu_item.copy()
            result['image_bytes'] = cached
            return result

        client = _get_client()

        print(f"🎨 Generating image for: {menu_item.get('name', 'Unknown')}")

        # A rate-limit rejection is expected under load, not an error worth
        # crashing over, so retry with backoff before giving up on the dish.
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                image_bytes = _call_image_model(client, prompt)

                if image_bytes:
                    store_image(key, image_bytes)

                    # Return the menu item with image data
                    result = menu_item.copy()
                    result['image_bytes'] = image_bytes
                    print(f"✅ Successfully generated image for: {menu_item.get('name', 'Unknown')}")
                    return result

                print(f"❌ Failed to generate image for: {menu_item.get('name', 'Unknown')}")
                return None

            except Exception as e:
                print(f"❌ Attempt {attempt}/{MAX_ATTEMPTS} failed for {menu_item.get('name', 'Unknown')}: {e}")
                # A 404 means the model is gone, not busy. Retrying it just burns
                # time and prints the same error three times.
                if '404' in str(e) or 'NOT_FOUND' in str(e):
                    print(f"❌ Model {IMAGE_MODEL} is unavailable; not retrying.")
                    return None
                if attempt < MAX_ATTEMPTS:
                    time.sleep(RETRY_SLEEP_SECONDS * attempt)

        print(f"❌ Giving up on: {menu_item.get('name', 'Unknown')} after {MAX_ATTEMPTS} attempts")
        return None

    except Exception as e:
        print(f"❌ Error generating image for {menu_item.get('name', 'Unknown')}: {e}")
        return None


def generate_images_for_menu(menu_items: list, restaurant_style: str = "", on_progress: Optional[Callable] = None) -> list:
    """
    Generates images for all menu items concurrently.

    Args:
        menu_items: List of menu item dictionaries
        restaurant_style: Style string injected into every prompt for visual consistency
        on_progress: Optional callback called with (completed_count, total_count, item_name)
                     after each image completes

    Returns:
        List of menu item dictionaries with image data
    """
    if not menu_items:
        return []

    total = len(menu_items)
    print(f"🚀 Starting image generation for {total} menu items (style: '{restaurant_style}')...")

    successful_results = []
    completed = 0

    # 6 workers against a 100 requests/minute limit. This was 3 when the app used
    # Imagen 4 Fast, which allowed only 10 a minute.
    with ThreadPoolExecutor(max_workers=6) as executor:
        # Submit all tasks, passing restaurant_style to each
        future_to_item = {
            executor.submit(generate_image, item, restaurant_style): item
            for item in menu_items
        }

        # Collect results as they complete
        for future in as_completed(future_to_item):
            completed += 1
            item = future_to_item[future]

            try:
                result = future.result()
                if result:
                    successful_results.append(result)
            except Exception as e:
                print(f"❌ Error generating image for {item.get('name', 'Unknown')}: {e}")

            if on_progress:
                on_progress(completed, total, item.get('name', 'Unknown'))

    print(f"✅ Successfully generated {len(successful_results)} out of {total} images")
    return successful_results