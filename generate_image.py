import os
import random
import requests
from PIL import Image, ImageDraw, ImageFont
import textwrap
import csv
from bs4 import BeautifulSoup

# -----------------------------
# CONFIGURATION
# -----------------------------
languages = {
    "EN": "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "MY": "https://ms.wikipedia.org/wiki/Kecerdasan_buatan",
    "CN": "https://zh.wikipedia.org/wiki/人工智能"  # Chinese Wikipedia
}

output_dir = "ocr_imagesdemo"
font_paths = {
    "EN": r"C:\Users\Aaron Lim\Dropbox\PC\Documents\1School\FYP\Language Font\Noto_Sans\NotoSans-Regular.ttf",
    "MY": r"C:\Users\Aaron Lim\Dropbox\PC\Documents\1School\FYP\Language Font\Noto_Sans\NotoSans-Regular.ttf",
    "CN": r"C:\Users\Aaron Lim\Dropbox\PC\Documents\1School\FYP\Language Font\Noto_Sans_SC\NotoSansSC-Regular.ttf"  # Chinese font
}
image_size = (800, 200)
font_size = 32
max_chars_per_line = 40
num_images_per_language = 50

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def fetch_wikipedia_text(url):
    """Fetch plain text from Wikipedia page paragraphs"""
    print(f"  → Fetching URL: {url}")
    try:
        # Wikipedia requires a User-Agent header
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        paragraphs = soup.find_all("p")
        text_data = []
        for p in paragraphs:
            text = p.get_text().strip()
            if len(text) > 50:
                text_data.append(text)
        print(f"  → Found {len(text_data)} paragraphs")
        return text_data
    except Exception as e:
        print(f"  ❌ ERROR fetching Wikipedia: {e}")
        return []

def load_font(font_path, size):
    """Load font, fallback to default if not found"""
    if not os.path.exists(font_path):
        print(f"  ⚠ Font file NOT FOUND: {font_path}")
        print(f"  → Using default font instead")
        return ImageFont.load_default()
    
    try:
        font = ImageFont.truetype(font_path, size)
        print(f"  ✅ Loaded font: {font_path}")
        return font
    except Exception as e:
        print(f"  ⚠ Cannot load font '{font_path}': {e}")
        return ImageFont.load_default()

def create_text_image(text, font_path, output_path):
    """Render text into an image and save"""
    try:
        img = Image.new("RGB", image_size, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        font = load_font(font_path, font_size)
        
        # Wrap text
        wrapped_text = textwrap.fill(text, width=max_chars_per_line)
        draw.text((10, 10), wrapped_text, font=font, fill=(0, 0, 0))
        img.save(output_path)
        print(f"  ✅ Saved: {output_path}")
        return True
    except Exception as e:
        print(f"  ❌ ERROR creating image: {e}")
        return False

# -----------------------------
# MAIN SCRIPT
# -----------------------------
print("="*60)
print("Starting OCR Image Generator")
print("="*60)

os.makedirs(output_dir, exist_ok=True)
print(f"✅ Output directory: {output_dir}")

label_file = os.path.join(output_dir, "labels.csv")

with open(label_file, mode="w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "language"])
    
    for lang_code, url in languages.items():
        print(f"\n{'='*60}")
        print(f"Processing: {lang_code}")
        print(f"{'='*60}")
        
        texts = fetch_wikipedia_text(url)
        
        if not texts:
            print(f"  ⚠ No text data found for {lang_code}, skipping...")
            continue
        
        random.shuffle(texts)
        texts = texts[:num_images_per_language]
        print(f"  → Selected {len(texts)} texts to generate")
        
        font_path = font_paths.get(lang_code, "")
        if not font_path:
            print(f"  ⚠ No font path defined for {lang_code}")
        
        for idx, text in enumerate(texts):
            filename = f"{lang_code}_text_{idx+1:04d}.png"
            output_path = os.path.join(output_dir, filename)
            
            print(f"\n  [{idx+1}/{len(texts)}] Generating {filename}...")
            print(f"  Text preview: {text[:80]}...")
            
            success = create_text_image(text, font_path, output_path)
            if success:
                writer.writerow([filename, lang_code])

print("\n" + "="*60)
print("✅ Done! Check the output above for any errors.")
print("="*60)