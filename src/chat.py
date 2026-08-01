import os
import json
from google import genai
from google.genai import types

# gemini-2.5-flash returns 404 'no longer available to new users' on keys
# issued recently, so this pins a currently served model.
TEXT_MODEL = 'gemini-3.6-flash'


CHAT_SYSTEM_INSTRUCTION = """You are a friendly and knowledgeable menu assistant for a restaurant.
You have been given the full menu as JSON data. Your job is to help customers with questions about
the menu, such as dietary restrictions, recommendations, allergens, ingredients, nutritional info,
or anything else related to the dishes available.

Be concise, warm, and helpful. Reference specific dishes by name when relevant.
If asked about something not on the menu, politely say it is not available."""


class MenuChatAgent:
    """
    A conversational agent that answers questions about a restaurant menu using Gemini.

    Initializes a Gemini chat session with the full menu JSON silently injected
    as system context, so the user never needs to paste or upload data again.
    """

    def __init__(self, menu_items: list, restaurant_style: str = ""):
        """
        Args:
            menu_items: List of menu item dicts (the full extracted menu).
            restaurant_style: Restaurant vibe string for additional context.
        """
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Google API Key not found.")

        self._client = genai.Client(api_key=api_key)

        # Build the system prompt with menu data silently injected
        menu_json = json.dumps(menu_items, indent=2, ensure_ascii=False)
        style_context = f"\nRestaurant style/vibe: {restaurant_style}" if restaurant_style else ""
        system_prompt = (
            f"{CHAT_SYSTEM_INSTRUCTION}{style_context}\n\n"
            f"Here is the full menu in JSON format:\n```json\n{menu_json}\n```"
        )

        self._chat = self._client.chats.create(
            model=TEXT_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
            )
        )

    def ask(self, question: str) -> str:
        """
        Send a question and return the assistant's response.

        Args:
            question: The user's question about the menu.

        Returns:
            str: The agent's response text.
        """
        try:
            response = self._chat.send_message(question)
            return response.text
        except Exception as e:
            return f"Sorry, I couldn't process that question. ({e})"
