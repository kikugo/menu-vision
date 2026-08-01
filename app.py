import streamlit as st
import io
import json
import os
from pathlib import Path
from collections import Counter
from PIL import Image
from dotenv import load_dotenv

from src.vision import extract_menu_items_from_image, extract_menu_items_from_text, stream_menu_items
from src.imaging import generate_images_for_menu, generate_image
from src.chat import MenuChatAgent
from concurrent.futures import ThreadPoolExecutor, as_completed

# Demo deployment guards. Same DEMO_MODE convention as unified-rag.
DEMO_MODE = os.getenv("DEMO_MODE", "false").strip().lower() in {"1", "true", "yes"}
MAX_DISHES_PER_SESSION = int(os.getenv("MAX_DISHES_PER_SESSION", "6"))

# Load environment variables from .env file
load_dotenv()

def initialize_ui():
    """Initializes the Streamlit page configuration and UI elements."""
    st.set_page_config(
        page_title="Menu-Vision", 
        page_icon="📸", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("📸 Menu-Vision")
    st.markdown("""
    Turns a restaurant menu into a set of dish photos.

    Upload a photo of the menu or paste the text. Gemini 3.5 Flash Lite reads each dish,
    then Gemini 3.1 Flash Lite Image generates a photo for it.
    """)
    
    st.divider()

def display_menu_item(item, col):
    """Display a single menu item in a card format with proper error handling."""
    with col:
        with st.container():
            # Create a styled card
            st.markdown(
                """
                <style>
                .menu-card {
                    border: 1px solid #e0e0e0;
                    border-radius: 10px;
                    padding: 15px;
                    margin: 10px 0;
                    background: white;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .menu-item-name {
                    font-size: 1.2em;
                    font-weight: bold;
                    color: #2e2e2e;
                    margin: 10px 0 5px 0;
                }
                .menu-item-description {
                    font-size: 0.9em;
                    color: #666;
                    margin: 5px 0;
                    font-style: italic;
                }
                .menu-item-price {
                    font-size: 1.1em;
                    font-weight: bold;
                    color: #ff6b35;
                    margin: 5px 0;
                }
                .tags-container {
                    margin: 10px 0;
                }
                .tag {
                    display: inline-block;
                    background-color: #e8f5e9;
                    color: #388e3c;
                    padding: 3px 8px;
                    border-radius: 15px;
                    font-size: 0.8em;
                    margin-right: 5px;
                    margin-bottom: 5px;
                }
                .no-image-placeholder {
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    border-radius: 8px;
                    padding: 40px 20px;
                    text-align: center;
                    color: #666;
                    font-style: italic;
                }
                .nutrition-container {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 6px;
                    margin: 8px 0;
                }
                .nutrition-pill {
                    display: inline-flex;
                    align-items: center;
                    gap: 3px;
                    background-color: #f0f4ff;
                    color: #3b5bdb;
                    padding: 3px 9px;
                    border-radius: 20px;
                    font-size: 0.78em;
                    font-weight: 600;
                    border: 1px solid #c5d2f6;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            
            # Display image with error handling
            if item.get('image_bytes'):
                try:
                    image = Image.open(io.BytesIO(item['image_bytes']))
                    st.image(image, width="stretch", caption="")
                except Exception as e:
                    st.markdown(
                        '<div class="no-image-placeholder">🖼️ Image Error<br><small>Could not load generated image</small></div>',
                        unsafe_allow_html=True
                    )
            else:
                # Placeholder for failed image generation
                st.markdown(
                    '<div class="no-image-placeholder">🎨 Image Generation Failed<br><small>Try regenerating or check API keys</small></div>',
                    unsafe_allow_html=True
                )
            
            # Display item name (required field)
            name = item.get('name', 'Unknown Dish')
            st.markdown(f'<div class="menu-item-name">{name}</div>', unsafe_allow_html=True)
            
            # Display description if available and not empty
            description = item.get('description', '').strip()
            if description:
                st.markdown(f'<div class="menu-item-description">{description}</div>', unsafe_allow_html=True)
            
            # Display ingredients if available
            ingredients = item.get('ingredients', [])
            if ingredients:
                st.markdown(f'<div class="menu-item-description"><i>Ingredients: {", ".join(ingredients)}</i></div>', unsafe_allow_html=True)

            # Display nutrition pills if available
            calories = item.get('estimated_calories')
            protein = item.get('protein_g')
            carbs = item.get('carbs_g')
            fat = item.get('fat_g')
            if any(v is not None for v in [calories, protein, carbs, fat]):
                pills = []
                if calories is not None: pills.append(f'🔥 {calories} kcal')
                if protein is not None:  pills.append(f'💪 {protein}g protein')
                if carbs is not None:    pills.append(f'🌾 {carbs}g carbs')
                if fat is not None:      pills.append(f'🫒 {fat}g fat')
                pills_html = "".join([f'<span class="nutrition-pill">{p}</span>' for p in pills])
                st.markdown(f'<div class="nutrition-container">{pills_html}</div>', unsafe_allow_html=True)

            # Display tags if available
            tags = item.get('tags', [])
            if tags:
                tags_html = "".join([f'<span class="tag">{tag}</span>' for tag in tags])
                st.markdown(f'<div class="tags-container">{tags_html}</div>', unsafe_allow_html=True)

            # Display price if available and not empty
            price = item.get('price', '').strip()
            if price:
                st.markdown(f'<div class="menu-item-price">{price}</div>', unsafe_allow_html=True)

def _get_top_tags(menu_items, max_tags=6):
    """Extract the most common tags from menu items for search suggestions."""
    all_tags = []
    for item in menu_items:
        all_tags.extend(item.get('tags', []))
    # Get the most common tags, title-cased
    tag_counts = Counter(tag.lower() for tag in all_tags)
    return [tag.title() for tag, _ in tag_counts.most_common(max_tags)]

SAMPLE_DIR = Path(__file__).parent / "examples" / "sample"


def _display_sample_menu():
    """Show a menu generated earlier, before the visitor uploads anything.

    The images are committed, so the common visit costs no API calls at all.
    Labelled as a sample on purpose: presenting cached output as a live result
    would be dishonest, and the caption is how a visitor knows the difference.
    """
    manifest_path = SAMPLE_DIR / "manifest.json"
    if not manifest_path.exists():
        return

    payload = json.loads(manifest_path.read_text())
    items = []
    for entry in payload.get("items", []):
        image_path = SAMPLE_DIR / entry.get("image", "")
        if not image_path.exists():
            continue
        item = dict(entry)
        item["image_bytes"] = image_path.read_bytes()
        items.append(item)

    if not items:
        return

    st.caption(
        "Example menu, generated earlier and saved. Upload a menu or paste text "
        "above to generate new images."
    )
    display_menu_grid(items)


def display_menu_grid(menu_items_with_images):
    """Display menu items in a responsive grid layout."""
    if not menu_items_with_images:
        st.warning("No menu items to display.")
        return

    # --- Top bar: title, item count, and Start Over ---
    top_left, top_right = st.columns([3, 1])
    with top_left:
        st.subheader(f"🍽️ Visual Menu")
    with top_right:
        st.markdown(
            f"<div style='display:flex; align-items:center; justify-content:flex-end; gap:12px; padding-top:8px;'>"
            f"<span style='color:#666; font-size:0.9em;'>{len(menu_items_with_images)} items</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        if st.button("🔄 Start Over", width="stretch"):
            st.session_state.clear()
            st.rerun()

    # --- Search bar + Ask the Menu button ---
    suggested_tags = _get_top_tags(menu_items_with_images)

    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    if 'show_chat' not in st.session_state:
        st.session_state.show_chat = False

    if suggested_tags:
        tag_cols = st.columns(len(suggested_tags))
        for i, tag in enumerate(suggested_tags):
            if tag_cols[i].button(tag, width="stretch"):
                st.session_state.search_query = tag

    search_col, chat_btn_col = st.columns([5, 1])
    with search_col:
        st.session_state.search_query = st.text_input(
            "Search by dish, ingredient, or flavor...",
            value=st.session_state.search_query,
            placeholder="e.g., 'Spicy', 'Tomatoes', 'Pasta'",
            label_visibility="collapsed"
        ).lower()
    with chat_btn_col:
        if st.button("💬 Ask the Menu", width="stretch"):
            st.session_state.show_chat = not st.session_state.get('show_chat', False)

    # --- Filter menu items ---
    if st.session_state.search_query:
        search_query = st.session_state.search_query
        filtered_items = [
            item for item in menu_items_with_images
            if search_query in item.get('name', '').lower() or \
               search_query in item.get('description', '').lower() or \
               any(search_query in ingredient.lower() for ingredient in item.get('ingredients', [])) or \
               any(search_query in tag.lower() for tag in item.get('tags', []))
        ]
        if not filtered_items:
            st.info(f"No dishes found matching '{search_query}'.")
            return
        menu_items_to_display = filtered_items
    else:
        menu_items_to_display = menu_items_with_images

    # --- Grid ---
    num_cols = min(3, len(menu_items_to_display))
    if num_cols == 0:
        return
    cols = st.columns(num_cols)

    for i, item in enumerate(menu_items_to_display):
        col_index = i % num_cols
        display_menu_item(item, cols[col_index])

def process_menu(menu_items, st_container):
    """
    Validates menu items and then generates images with progress feedback.
    """
    with st_container:
        # 1. Validate menu items
        is_valid, message = validate_menu_items(menu_items)
        if not is_valid:
            st.error(message)
            return None
        
        st.success(message)

        # Imagen 4 Fast allows 70 images a day on this tier and one image is one
        # dish, so a single 40 dish menu would eat most of it. Cap the demo and say
        # so out loud: silently returning six of forty dishes reads as a bug.
        if DEMO_MODE and len(menu_items) > MAX_DISHES_PER_SESSION:
            st.warning(
                f"Demo limit: generating the first {MAX_DISHES_PER_SESSION} of "
                f"{len(menu_items)} dishes. Clone the repo to generate a full menu."
            )
            menu_items = menu_items[:MAX_DISHES_PER_SESSION]

        # 2. Generate images with per-item progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        def on_progress(completed, total, item_name):
            progress_bar.progress(completed / total)
            status_text.text(f"Generated {completed}/{total}: {item_name}")

        visual_menu = generate_images_for_menu(menu_items, on_progress=on_progress)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        if visual_menu:
            st.success(f"Generated {len(visual_menu)} dish images.")
            return visual_menu
        else:
            st.error("Failed to generate any images. Please check your API keys and try again.")
            return None

def validate_menu_items(menu_items):
    """Validate extracted menu items and provide user feedback."""
    if not menu_items:
        return False, "No menu items were extracted. Please try a clearer image or check your text format."
    
    # Check for valid items
    valid_items = []
    issues = []
    
    for i, item in enumerate(menu_items, 1):
        if not item.get('name', '').strip():
            issues.append(f"Item {i}: Missing name")
        elif not item.get('prompt', '').strip():
            issues.append(f"Item {i} ({item['name']}): Missing image prompt")
        else:
            valid_items.append(item)
    
    if not valid_items:
        return False, f"No valid menu items found. Issues: {'; '.join(issues)}"
    
    if issues:
        st.warning(f"Some items have issues but will proceed with {len(valid_items)} valid items: {'; '.join(issues)}")
    
    return True, f"Successfully extracted {len(valid_items)} valid menu items!"

def main():
    """Main application function."""
    initialize_ui()

    import os

    # Sidebar for input options
    with st.sidebar:
        # --- API Key Status ---
        api_key = os.getenv('GOOGLE_API_KEY') or st.session_state.get('_runtime_api_key', '')
        st.markdown("### 🔑 API Key")
        if api_key:
            st.success("✅ Gemini key detected", icon=None)
        else:
            st.error("❌ Key not found", icon=None)
            new_key = st.text_input(
                "Enter your Google API key:",
                type="password",
                placeholder="AIza...",
                help="Your key is stored in session memory only — not saved to disk."
            )
            if new_key:
                os.environ['GOOGLE_API_KEY'] = new_key
                st.session_state['_runtime_api_key'] = new_key
                st.rerun()

        st.divider()

        st.header("📋 Menu Input")
        input_method = st.radio(
            "Choose input method:",
            ["Upload Image", "Paste Text"],
            index=0
        )

        if input_method == "Upload Image":
            st.subheader("📸 Upload Menu Pages")
            uploaded_files = st.file_uploader(
                "Upload one or more menu page images...",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                help="Upload up to 200 MB per file. Multiple pages supported."
            )

            if uploaded_files:
                images = [Image.open(f) for f in uploaded_files]
                # Show thumbnails in a compact row
                thumb_cols = st.columns(min(len(images), 4))
                for i, img in enumerate(images):
                    thumb_cols[i % 4].image(img, caption=f"Page {i+1}", width="stretch")
                
                if st.button("🍽️ Create Visual Menu", type="primary"):
                    st.session_state.clear()
                    st.session_state.processing = True

                    progress_bar = st.progress(0, text="Reading menu...")
                    cols = st.columns(3)
                    placeholders = []
                    futures = []
                    items_collected = []
                    restaurant_style = ""

                    with ThreadPoolExecutor(max_workers=5) as executor:
                        from src.vision import SYSTEM_PROMPT as _SP
                        # Pass all page images in a single API call for full menu context
                        contents = [_SP] + images
                        for style_or_empty, item in stream_menu_items(contents):
                            if item is None:  # First yield: restaurant_style
                                restaurant_style = style_or_empty
                                if style_or_empty == "ERROR":
                                    st.error("Failed to read menu stream.")
                                    break
                                continue

                            items_collected.append(item)
                            idx = len(items_collected) - 1
                            col = cols[idx % 3]
                            ph = col.empty()
                            ph.markdown(f"**{item.get('name', '')}** — ⏳ generating...")
                            placeholders.append((ph, item))

                            # Fire image gen immediately for this item
                            f = executor.submit(generate_image, item, restaurant_style)
                            futures.append((f, ph, item, idx))

                        # Collect results as they finish
                        total = len(futures)
                        done = 0
                        visual_menu = [None] * total
                        for future, ph, item, idx in futures:
                            try:
                                result = future.result()
                                if result:
                                    visual_menu[idx] = result
                                    image_obj = __import__('PIL.Image', fromlist=['Image']).open(__import__('io').BytesIO(result['image_bytes']))
                                    ph.image(image_obj, width="stretch", caption=item.get('name', ''))
                                else:
                                    visual_menu[idx] = item
                                    ph.warning(f"⚠️ No image for {item.get('name', '')}")
                            except Exception as e:
                                visual_menu[idx] = item
                                ph.warning(f"⚠️ Error: {e}")
                            done += 1
                            progress_bar.progress(done / max(total, 1), text=f"Generated {done}/{total} images")

                    st.session_state.restaurant_style = restaurant_style
                    st.session_state.menu_items = items_collected
                    st.session_state.visual_menu = [v for v in visual_menu if v]
                    progress_bar.empty()

        # User input for pasting menu text
        elif input_method == "Paste Text":
            st.subheader("📝 Menu Text")
            menu_text = st.text_area(
                "Paste your menu text here:",
                height=200,
                placeholder="Example:\nMargherita Pizza - Tomato, Mozzarella, Basil - $15"
            )
            
            if st.button("🍽️ Create Visual Menu", type="primary"):
                if menu_text.strip():
                    st.session_state.clear()
                    st.session_state.processing = True

                    progress_bar = st.progress(0, text="Reading menu...")
                    cols_text = st.columns(3)
                    futures_text = []
                    items_text = []
                    restaurant_style = ""

                    with ThreadPoolExecutor(max_workers=5) as executor:
                        text_contents = f"{__import__('src.vision', fromlist=['SYSTEM_PROMPT']).SYSTEM_PROMPT}\n\nHere is the menu text to analyze:\n{menu_text}"
                        for style_or_empty, item in stream_menu_items(text_contents):
                            if item is None:
                                restaurant_style = style_or_empty
                                if style_or_empty == "ERROR":
                                    st.error("Failed to read menu stream.")
                                    break
                                continue

                            items_text.append(item)
                            idx = len(items_text) - 1
                            ph = cols_text[idx % 3].empty()
                            ph.markdown(f"**{item.get('name', '')}** — ⏳ generating...")

                            f = executor.submit(generate_image, item, restaurant_style)
                            futures_text.append((f, ph, item, idx))

                        total_t = len(futures_text)
                        done_t = 0
                        visual_menu_text = [None] * total_t
                        for future, ph, item, idx in futures_text:
                            try:
                                result = future.result()
                                if result:
                                    visual_menu_text[idx] = result
                                    image_obj = __import__('PIL.Image', fromlist=['Image']).open(__import__('io').BytesIO(result['image_bytes']))
                                    ph.image(image_obj, width="stretch", caption=item.get('name', ''))
                                else:
                                    visual_menu_text[idx] = item
                                    ph.warning(f"⚠️ No image for {item.get('name', '')}")
                            except Exception as e:
                                visual_menu_text[idx] = item
                                ph.warning(f"⚠️ Error: {e}")
                            done_t += 1
                            progress_bar.progress(done_t / max(total_t, 1), text=f"Generated {done_t}/{total_t} images")

                    st.session_state.restaurant_style = restaurant_style
                    st.session_state.menu_items = items_text
                    st.session_state.visual_menu = [v for v in visual_menu_text if v]
                    progress_bar.empty()
                else:
                    st.warning("Please enter some menu text first.")

    if "visual_menu" not in st.session_state:
        _display_sample_menu()

    if "visual_menu" in st.session_state:
        display_menu_grid(st.session_state.visual_menu)

        # --- Inline Chat Panel (toggled by 'Ask the Menu' button in header) ---
        if st.session_state.get('menu_items') and st.session_state.get('show_chat', False):
            st.divider()
            st.markdown("#### 💬 Ask the Menu")

            if 'chat_agent' not in st.session_state:
                try:
                    st.session_state.chat_agent = MenuChatAgent(
                        menu_items=st.session_state.menu_items,
                        restaurant_style=st.session_state.get('restaurant_style', '')
                    )
                    st.session_state.chat_history = []
                except Exception as e:
                    st.error(f"Could not initialize chat agent: {e}")

            if 'chat_agent' in st.session_state:
                for msg in st.session_state.get('chat_history', []):
                    with st.chat_message(msg['role']):
                        st.markdown(msg['content'])

                user_q = st.chat_input("Ask anything about the menu…")
                if user_q:
                    st.session_state.chat_history.append({'role': 'user', 'content': user_q})
                    with st.chat_message('user'):
                        st.markdown(user_q)
                    with st.chat_message('assistant'):
                        with st.spinner(""):
                            reply = st.session_state.chat_agent.ask(user_q)
                        st.markdown(reply)
                    st.session_state.chat_history.append({'role': 'assistant', 'content': reply})

if __name__ == "__main__":
    main()