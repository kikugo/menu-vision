# Menu Vision

Turns a restaurant menu into a set of dish photos. Upload a photo of the menu or paste the text, and it reads each dish, then generates an image for it.

The use case I had in mind is a restaurant that needs images for a delivery listing or a website and does not want to pay for a photo shoot. A menu with 40 dishes takes minutes instead of a day.

Opening the app shows an example menu that was generated earlier and committed to the repo, so looking around costs no API calls. Uploading a menu generates new images. Generated images are cached on disk by prompt, so the same dish is never paid for twice.

A note on models: this used Imagen 4 Fast and Gemini 2.5 Flash until August 2026, when both stopped being served to newly issued API keys and started returning 404. It now uses Gemini 3.1 Flash Image and Gemini 3.6 Flash, which also lifted the image quota from 70 a day to 1000.

![Python](https://img.shields.io/badge/Python-3.8+-green) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)

## Features

- **Menu reading**: Gemini 3.6 Flash does the OCR and extracts each dish's name, description and price
- **Image generation**: Gemini 3.1 Flash Image generates one photo per dish
- **Style consistency**: Gemini also infers the restaurant's visual style from the menu, and that description gets appended to every image prompt so the set of photos looks like one shoot instead of forty unrelated ones
- **Concurrent generation**: images generate in parallel with a thread pool, and OCR streams back to the UI as it runs instead of waiting for the full response
- **Ask the menu**: a chatbot answers questions about dietary info, recommendations, and pairings, using the extracted menu text
- **Nutrition estimate**: calories, protein, carbs and fat per dish, estimated by the model rather than measured
- **Two ways in**: upload a menu photo or paste the text directly
- **Search**: filter dishes by name, description, ingredients, or tag

## Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/), a Python web UI framework
- **OCR & Menu Parsing**: [Google Gemini 3.6 Flash](https://deepmind.google/technologies/gemini/) via `google-genai` SDK
- **Image Generation**: [Google Gemini 3.1 Flash Image](https://deepmind.google/technologies/gemini/) via `google-genai` SDK
- **Language**: Python 3.8+

> **Note**: This project uses a single API provider (Google) for both OCR and image generation, so only one API key is needed.

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/kikugo/visual-menu-ai.git
cd menu_vision_ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up API Key
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY="your_gemini_api_key_here"
```

**Get your API key:**
- 🔗 [Google Gemini API Key](https://aistudio.google.com/app/apikey) (free tier available)

### 4. Run the Application
```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

## How It Works

1. **Upload/Input**: Choose to upload a menu image or paste menu text
2. **Extract**: Gemini reads the menu, structures each item, and generates tags for it
3. **Generate**: Gemini 3.1 Flash Image generates a photo for each dish, concurrently
4. **Display**: the menu renders as a responsive grid of dishes and photos
5. **Search**: the search bar filters dishes by name, description, ingredients, or tags

### Menu Item Structure
Each extracted item includes:
```json
{
  "name": "Dish name",
  "description": "Brief description", 
  "price": "Price string",
  "ingredients": ["ingredient1", "ingredient2"],
  "tags": ["tag1", "tag2"],
  "prompt": "Custom image generation prompt"
}
```

## Testing

The `tests` folder contains scripts to test the core API integrations independently from the main application.

### Prerequisites

1.  Make sure you have set up your `.env` file in the project root with your API key.
2.  Ensure all dependencies are installed: `pip install -r requirements.txt`

### Running the Tests

```bash
cd tests

# Test Google Gemini API (OCR and menu extraction)
python test_gemini.py

# Test Imagen 4 API (Image generation)
python test_imagen.py
```

### Expected Results

-   **Gemini Test**: prints a pass message once all Gemini tests pass.
    -   *Tests*: Basic text generation, menu extraction from text, and menu extraction from an image (if `example.jpeg` or similar is present in examples folder).
-   **Imagen Test**: prints a pass message once all Imagen tests pass.
    -   *Tests*: API connectivity and generation of several test images. Generated images are saved in the `tests` folder for you to review.

If any tests fail, double-check your API key in the `.env` file and your internet connection.

## Project Structure

```
menu_vision_ai/
├── app.py                  # Main Streamlit application
├── src/
│   ├── __init__.py        # Makes src a Python package
│   ├── vision.py          # Gemini OCR & menu extraction  
│   └── imaging.py         # Imagen 4 image generation
├── tests/
│   ├── test_gemini.py     # Gemini API tests
│   └── test_imagen.py     # Imagen 4 API tests
├── examples/
│   ├── sample_menu.txt    # Sample menu text for input
│   ├── example.jpeg       # Sample menu images
│   ├── example2.jpeg
│   └── example3.jpeg
├── requirements.txt        # Python dependencies
├── .env                    # API key (create this yourself)
├── .gitignore              # Git ignore rules
├── LICENSE                 # Project license
└── README.md               # This file
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google Gemini / Imagen API key | ✅ Yes |

### Customization

- **Image Aspect Ratio**: Modify in `src/imaging.py` (default: 1:1 for food photos)
- **Menu Prompt Template**: Update `SYSTEM_PROMPT` in `src/vision.py`
- **UI Layout**: Customize grid columns in `app.py` `display_menu_grid()`
- **Concurrency**: Adjust `max_workers` in `src/imaging.py` (default: 5)

## Troubleshooting

### Common Issues

**"API Key not found"**
- Verify `.env` file exists in project root
- Check API key format and validity

**"Image generation failed"**  
- Confirm your Google API key has Imagen 4 access enabled
- Check internet connection
- Verify API service status at [Google AI Studio](https://aistudio.google.com/)

**"Menu extraction failed"**
- Try a clearer image or better formatted text
- Check Gemini API quota and limits

**PowerShell errors on Windows**
- Use `python -m streamlit run app.py` instead

### Debug Mode
Run tests to isolate issues:
```bash
cd tests
python test_gemini.py   # Test menu extraction
python test_imagen.py   # Test image generation
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`) 
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Andrej Karpathy](https://karpathy.bearblog.dev/vibe-coding-menugen/) for the original MenuGen inspiration
- [Nutlope/picMenu](https://github.com/Nutlope/picMenu) for the open-source reference
- Google for the Gemini and Imagen APIs

## Future Improvements

Ideas not yet built:

- **Semantic search**: keyword matching now, but converting menu items and the query into vector embeddings would let a search like "something hearty for a cold day" match soups and rich pasta instead of requiring those exact words.

- **Cuisine style detection**: identify the cuisine type (Italian, Mexican, Thai) and let users filter by it.

- **Menu analysis**: aggregated stats from the menu, such as the ratio of vegetarian to non-vegetarian dishes or the most common ingredients.