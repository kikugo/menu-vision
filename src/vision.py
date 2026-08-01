import os
import json
from google import genai
from google.genai import types

# gemini-2.5-flash returns 404 'no longer available to new users' on keys
# issued recently, so this pins a currently served model.
TEXT_MODEL = 'gemini-3.6-flash'

# Enhanced system prompt to guide the AI model in extracting menu items and generating image prompts
SYSTEM_PROMPT = """
You are MenuVision, an expert at reading restaurant menus and creating detailed food imagery prompts.

Your task is to:
1. Identify the overall aesthetic and style of the restaurant from the menu's design, name, and cuisine type
2. Analyze the menu image and extract each dish
3. Create a detailed image generation prompt for each dish
4. Return everything in a structured JSON format

For each menu item, extract:
- "name": The exact name of the dish from the menu
- "description": Brief description if available (empty string if none)
- "price": Price as string if available (empty string if none)
- "ingredients": A list of the main ingredients as strings. If none are listed, provide a reasonable guess based on the dish name.
- "tags": A list of relevant tags describing flavor profiles (e.g., spicy, sweet, savory), main ingredients (e.g., chicken, beef, vegetarian), and potential cuisine types (e.g., italian, mexican).
- "prompt": A detailed, appetizing image generation prompt
- "estimated_calories": An integer estimate of the total calories for one serving of this dish.
- "protein_g": An integer estimate of the protein in grams for one serving.
- "carbs_g": An integer estimate of the carbohydrates in grams for one serving.
- "fat_g": An integer estimate of the fat in grams for one serving.

For the image prompt, create a professional food photography description that includes:
- The dish name and key ingredients/components
- Presentation style (plated, served in bowl, etc.)
- Lighting (soft, natural, studio lighting)
- Camera angle (overhead, 45-degree, close-up)
- Background and styling details
- Adjectives that make the food look appetizing

Example image prompt format:
"Professional food photography of [dish name]. [Description of ingredients and preparation]. Beautifully plated on a white ceramic plate, shot from a 45-degree angle with soft natural lighting. Garnished elegantly, with a clean restaurant background. High-resolution, appetizing, and cinematic presentation."

Return ONLY a valid JSON object in this exact format:
{
  "restaurant_style": "A concise 5-10 word description of the restaurant's visual aesthetic and vibe (e.g., 'rustic Italian trattoria with warm earthy tones', 'sleek modern sushi bar with minimalist dark aesthetic', 'bright casual American diner with retro styling')",
  "items": [
    {
      "name": "Dish name",
      "description": "Brief description or empty string",
      "price": "Price or empty string",
      "ingredients": ["ingredient1", "ingredient2", "ingredient3"],
      "tags": ["flavor_tag", "ingredient_tag", "cuisine_tag"],
      "prompt": "Detailed image generation prompt",
      "estimated_calories": 650,
      "protein_g": 35,
      "carbs_g": 55,
      "fat_g": 28
    }
  ]
}

Important rules:
- Do NOT include price information in the image prompt
- Focus on making the food look as appetizing as possible
- Use professional food photography terminology
- If no description exists, leave as empty string
- If no price exists, leave as empty string
- Always include proper garnishing and presentation details in prompts
- Nutrition estimates should be realistic based on typical restaurant portion sizes
- Return ONLY the JSON object, no markdown formatting or additional text
"""

def _extract_menu_items(contents):
    """
    Internal helper to call the Gemini API and extract menu items.

    Args:
        contents: The contents to send to the Gemini API (list or string).

    Returns:
        tuple: (restaurant_style: str, items: list) or ("", []) on failure.
    """
    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")

        client = genai.Client(api_key=api_key)

        # Generate response using Gemini 3.6 Flash with structured JSON output
        response = client.models.generate_content(
            model=TEXT_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        # Parse the JSON response
        try:
            data = json.loads(response.text)

            # Validate wrapper structure
            if not isinstance(data, dict) or 'items' not in data or 'restaurant_style' not in data:
                raise ValueError("Response is not a valid wrapper object with 'restaurant_style' and 'items'")

            restaurant_style = data.get('restaurant_style', '')
            menu_items = data['items']

            if not isinstance(menu_items, list):
                raise ValueError("'items' is not a JSON array")

            required_fields = ['name', 'description', 'price', 'ingredients', 'tags', 'prompt',
                               'estimated_calories', 'protein_g', 'carbs_g', 'fat_g']
            for item in menu_items:
                if not all(field in item for field in required_fields):
                    raise ValueError(f"Missing required fields in menu item: {item}")

            return restaurant_style, menu_items

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Raw response: {response.text}")
            return "", []

    except Exception as e:
        print(f"Error extracting menu items: {e}")
        return "", []

def extract_menu_items_from_image(image_data):
    """
    Extracts menu items from an uploaded image using Google Gemini 3.6 Flash.

    Args:
        image_data: PIL Image object or image bytes

    Returns:
        tuple: (restaurant_style: str, items: list)
    """
    return _extract_menu_items([SYSTEM_PROMPT, image_data])

def extract_menu_items_from_text(menu_text):
    """
    Extracts menu items from plain text using Google Gemini 3.6 Flash.

    Args:
        menu_text: String containing the menu text

    Returns:
        tuple: (restaurant_style: str, items: list)
    """
    text_prompt = f"{SYSTEM_PROMPT}\n\nHere is the menu text to analyze:\n{menu_text}"
    return _extract_menu_items(text_prompt)

def stream_menu_items(contents):
    """
    Streams menu items one-by-one from Gemini as they are parsed.

    Uses Gemini's streaming API to yield individual menu item dicts the moment
    each one is complete, rather than waiting for the full response. This allows
    image generation to start for item #1 while OCR is still reading item #2.

    Args:
        contents: The contents to send to the Gemini API (list or string).

    Yields:
        tuple: First yield is (restaurant_style: str, None).
               Subsequent yields are ("", item: dict) for each parsed menu item.
               Yields ("ERROR", None) if streaming fails.
    """
    import re

    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Google API Key not found.")

        client = genai.Client(api_key=api_key)

        accumulated = ""
        restaurant_style_yielded = False
        restaurant_style = ""
        item_buffer = ""
        brace_depth = 0
        in_items_array = False

        for chunk in client.models.generate_content_stream(
            model=TEXT_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        ):
            if not chunk.text:
                continue
            accumulated += chunk.text

            # Extract restaurant_style as soon as it appears in the stream
            if not restaurant_style_yielded:
                style_match = re.search(r'"restaurant_style"\s*:\s*"([^"]*)"', accumulated)
                if style_match:
                    restaurant_style = style_match.group(1)
                    yield restaurant_style, None
                    restaurant_style_yielded = True

            # Once we locate the "items" array, track individual object boundaries
            if not in_items_array:
                if '"items"' in accumulated and '[' in accumulated.split('"items"', 1)[1]:
                    in_items_array = True
                    after_items = accumulated.split('"items"', 1)[1]
                    accumulated = after_items[after_items.index('[') + 1:]

            if in_items_array:
                for char in chunk.text:
                    if brace_depth == 0 and char == '{':
                        item_buffer = "{"
                        brace_depth = 1
                    elif brace_depth > 0:
                        item_buffer += char
                        if char == '{':
                            brace_depth += 1
                        elif char == '}':
                            brace_depth -= 1
                            if brace_depth == 0:
                                try:
                                    item = json.loads(item_buffer)
                                    yield "", item
                                except json.JSONDecodeError:
                                    pass
                                item_buffer = ""

    except Exception as e:
        print(f"Streaming error: {e}")
        yield "ERROR", None