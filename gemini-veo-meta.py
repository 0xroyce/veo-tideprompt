import json
import os
import time
import requests
import replicate
import re
from pydantic import BaseModel, Field, AnyUrl
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
# Make sure to create a .env file in the same directory as this script
# with OPENROUTER_API_KEY and REPLICATE_API_KEY
load_dotenv()

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Replicate configuration - handle both REPLICATE_API_KEY and REPLICATE_API_TOKEN
# If user has REPLICATE_API_KEY, set it as REPLICATE_API_TOKEN for the library
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY", "")
if REPLICATE_API_KEY and not os.getenv("REPLICATE_API_TOKEN"):
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_KEY


def sanitize_prompt(text: str, escape_for_json: bool = True) -> str:
    """
    Sanitize the prompt text to handle special characters that might break API calls.
    
    Args:
        text: The text to sanitize
        escape_for_json: Whether to escape quotes and backslashes for JSON safety
    """
    if not text:
        return text
    
    # Replace problematic characters
    replacements = {
        # Smart quotes to regular quotes
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        
        # Mathematical notation
        '~10^26': 'approximately 10 to the power of 26',
        '^': ' to the power of ',
        '≈': 'approximately',
        '±': 'plus or minus',
        '∞': 'infinity',
        
        # Other problematic characters
        '…': '...',
        '–': '-',
        '—': '-',
        
        # Remove or replace characters that can break JSON
        '\x00': '',  # null character
        '\r': ' ',   # carriage return
        '\t': ' ',   # tab
    }
    
    # Apply replacements
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Only escape for JSON if requested (not for JSON responses)
    if escape_for_json:
        # Escape backslashes and quotes for JSON safety
        text = text.replace('\\', '\\\\')
        text = text.replace('"', '\\"')
    
    # Remove any remaining control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    return text.strip()


class Shot(BaseModel):
    """Technical camera details for a specific clip."""

    composition: str = Field(
        ...,
        description="How the shot is framed and the lens used. Examples: 'Medium close-up, 35mm lens, deep focus, smooth gimbal', 'Extreme wide shot, 14mm lens, drone establishing shot with slow reveal', 'Dutch angle, 85mm portrait lens, handheld with intentional camera shake', 'Over-the-shoulder shot, 50mm lens, shallow depth of field'.",
    )
    camera_motion: str = Field(
        None,
        description="Describes the movement of the camera during the shot. Examples: 'slow dolly-in 60 cm', 'fast-paced tracking shot following the subject', 'static tripod shot with no movement', 'smooth jib arm crane movement from low to high', 'handheld push-in with slight wobble', 'circular dolly around subject'.",
    )
    frame_rate: str = Field(
        "24 fps",
        description="Frames per second, defining the motion look (24fps is cinematic). Examples: '24 fps', '60 fps for slow-motion effect', '120 fps for extreme slow motion', '12 fps for vintage or stop-motion feel'.",
    )
    film_grain: float = Field(
        None,
        description="Adds a stylistic film grain effect (0=none, higher values=more grain). Examples: 0.05, 0.15, 0.0, 0.3.",
    )
    camera: str = Field(
        ...,
        description="Camera lens, shot type, and equipment style for this clip. Examples: 'smooth gimbal 35mm', 'handheld iPhone with anamorphic lens adapter', 'RED camera on Steadicam rig', 'vintage 16mm film camera with prime lens'.",
    )


class Subject(BaseModel):
    """Describes the character's appearance and wardrobe within a specific clip."""

    description: str = Field(
        ...,
        description="A full, descriptive prompt of the character for this shot. Examples: 'Nyx Cipher — 27-year-old, 173 cm, toned-athletic build; deep-bronze skin glistening with water; jet-black slicked-back hair; almond hazel eyes behind mirrored sunglasses; small star tattoo behind right ear; wearing metallic-coral bikini and gold hoop earrings', 'Marcus Chen — 45-year-old chef, 180 cm, sturdy build; weathered hands from years of cooking; salt-and-pepper beard; warm brown eyes with laugh lines; wearing pristine white chef's coat with rolled sleeves', 'Luna-7 — ageless android appearing 25, 165 cm, sleek synthetic build; luminescent pale blue skin with circuit patterns; chrome-silver hair in geometric bob; violet LED eyes; wearing form-fitting matte black bodysuit with glowing accents'.",
    )
    wardrobe: str = Field(
        ...,
        description="The specific outfit worn in this clip. This can be based on the character's default_outfit. Examples: 'metallic-coral bikini, mirrored sunglasses, gold hoop earrings', 'weathered leather jacket, ripped jeans, combat boots, fingerless gloves', 'flowing emerald silk gown with intricate beadwork, diamond tiara', 'tactical gear with kevlar vest, utility belt, night vision goggles'.",
    )


class Scene(BaseModel):
    """Describes the setting and environment of the clip."""

    location: str = Field(
        ...,
        description="The physical place where the scene occurs. Examples: 'rooftop infinity pool overlooking a neon-tropic city skyline', 'abandoned Victorian mansion with overgrown ivy and broken windows', 'bustling Tokyo street market during cherry blossom season', 'underground speakeasy with dim lighting and jazz atmosphere'.",
    )
    time_of_day: str = Field(
        "mid-day",
        description="The time of day, which heavily influences lighting. Examples: 'mid-day', 'golden hour just before sunset', 'blue hour twilight', 'dead of night with only moonlight', 'early morning with soft dawn light', 'overcast afternoon'.",
    )
    environment: str = Field(
        ...,
        description="Specific details about the surroundings. Examples: 'sunlit pool water reflecting shifting patterns; floating dollar-sign inflatables', 'heavy rain creating puddles that reflect neon signs; steam rising from manholes', 'gentle snowfall accumulating on windowsills; warm light spilling from cozy windows', 'desert wind kicking up sand clouds; distant lightning illuminating cacti silhouettes'.",
    )


class VisualDetails(BaseModel):
    """Describes the actions and props within the clip."""

    action: str = Field(
        ...,
        description="What the character is physically doing in the scene. Examples: 'Nyx leans on pool edge and, on beat four, fans her hand cheekily toward camera as droplets sparkle in the air', 'Marcus carefully plates microgreens with tweezers, each movement precise and deliberate', 'Luna-7 interfaces with a holographic display, her fingers dancing through floating data streams', 'character parkours across rooftops, leaping between buildings with fluid grace'.",
    )
    props: str = Field(
        None,
        description="Objects that appear or are interacted with in the scene. Examples: 'floating dollar-sign inflatables', 'antique brass telescope pointing toward star-filled sky', 'holographic chess set with pieces that glow and float', 'vintage motorcycle with chrome details and leather saddlebags'.",
    )


class Cinematography(BaseModel):
    """Defines the artistic visual style for this clip."""

    lighting: str = Field(
        ...,
        description="Specific lighting direction for this shot. Examples: 'high-key mid-day sunlight with specular highlights on wet skin', 'dramatic chiaroscuro lighting with deep shadows and bright highlights', 'soft window light with gauzy curtains creating dappled patterns', 'neon-lit night scene with colorful reflections on wet pavement', 'candlelit interior with warm, flickering ambiance'.",
    )
    tone: str = Field(
        ...,
        description="The intended mood and feeling of the clip. Examples: 'vibrant, playful, confident', 'dark, suspenseful, and mysterious', 'warm, nostalgic, and sentimental', 'ethereal, dreamlike, and surreal', 'gritty, intense, and raw'.",
    )
    color_grade: str = Field(
        ...,
        description="The color correction and mood for this clip. Examples: 'hyper-saturated neon-tropic (hot-pink, aqua, tangerine)', 'desaturated, gritty, and cool-toned for a noir look', 'warm, golden tones to evoke nostalgia', 'high-contrast black and white with selective color pops', 'teal and orange blockbuster color scheme'.",
    )


class AudioTrack(BaseModel):
    """Defines the sound elements specific to this clip."""

    lyrics: Optional[str] = Field(
        None,
        description="The lyrics to be lip-synced or heard. Examples: 'Splash-cash, bling-blap—pool water pshh! Charts skrrt! like my wave, hot tropics whoosh!', 'In the silence of the ancient halls, whispers of forgotten souls call', 'Dancing through the neon lights, city never sleeps at night', 'Breaking chains of yesterday, finding strength to walk away'.",
    )
    emotion: Optional[str] = Field(
        None,
        description="The emotional tone of the vocal performance. Examples: 'confident, tongue-in-cheek', 'somber and melancholic', 'energetic and joyful', 'haunting and ethereal', 'aggressive and defiant', 'tender and vulnerable'.",
    )
    flow: Optional[str] = Field(
        None,
        description="The rhythm and cadence of the lyrical delivery (especially for rap). Examples: 'double-time for first bar, brief half-time tag', 'slow, spoken-word style with dramatic pauses', 'melodic and sing-song with flowing transitions', 'staccato rapid-fire delivery', 'syncopated rhythm with off-beat emphasis'.",
    )
    wave_download_url: Optional[AnyUrl] = Field(
        None,
        description="A URL to a pre-existing audio file for this clip (if available).",
    )
    youtube_reference: Optional[AnyUrl] = Field(
        None,
        description="A URL to a YouTube video as a reference for style or content.",
    )
    audio_base64: Optional[str] = Field(
        None,
        description="A base64 encoded string of the audio data, for embedding it directly.",
    )
    # -- Fields from former AudioDefaults --
    format: str = Field(
        "wav",
        description="The desired audio file format. Examples: 'wav', 'mp3', 'flac', 'aac'.",
    )
    sample_rate_hz: int = Field(
        48000,
        description="The audio quality in Hertz, affecting fidelity. Examples: 48000, 44100, 96000, 192000.",
    )
    channels: int = Field(
        2,
        description="The number of audio channels. Examples: 2 (stereo), 1 (mono), 6 (5.1 surround), 8 (7.1 surround).",
    )
    style: str = Field(
        None,
        description="Describes the musical genre, tempo, and elements for this track. Examples: 'trap-pop rap, 145 BPM, swung hats, sub-bass', 'orchestral score with sweeping strings and dramatic percussion, 60 BPM', 'lo-fi hip hop, 80 BPM, jazzy chords, vinyl crackle', 'synthwave with arpeggiated basslines and retro drums, 120 BPM'.",
    )


class Dialogue(BaseModel):
    """Defines the spoken lines and how they are presented."""

    character: str = Field(
        ...,
        description="The character who is speaking. Examples: 'Nyx Cipher', 'The Mysterious Stranger', 'AI System Voice', 'Narrator'.",
    )
    line: str = Field(
        ...,
        description="The exact line of dialogue or lyrics. Examples: 'Splash-cash, bling-blap—pool water pshh! Charts skrrt! like my wave, hot tropics whoosh!', 'The memories are all that remain of what we once were', 'Access granted. Welcome to the future', 'In a world where nothing is as it seems...'.",
    )
    subtitles: bool = Field(
        default=False,
        description="A boolean to determine if subtitles should be rendered for this line. Subtitles should always be false. Never add subtitles to the video.",
    )


class Performance(BaseModel):
    """Controls for the character's animated performance in this clip."""

    mouth_shape_intensity: Optional[float] = Field(
        None,
        description="Clip-specific override for lip-sync exaggeration (0=subtle, 1=exaggerated). Examples: 0.85, 0.3, 1.0, 0.1.",
    )
    eye_contact_ratio: Optional[float] = Field(
        None,
        description="Clip-specific override for how often the character looks at the camera. Examples: 0.7, 0.1, 1.0, 0.5.",
    )


# -- Main Clip Model --


class Clip(BaseModel):
    """Defines a single video segment or shot."""

    id: str = Field(
        ...,
        description="A unique identifier for this specific clip. Examples: 'S1_SplashCash', 'Forest_Intro_001', 'Cyberpunk_Market_Scene_3B', 'Chase_Sequence_Final'.",
    )
    shot: Shot
    subject: Subject
    scene: Scene
    visual_details: VisualDetails
    cinematography: Cinematography
    audio_track: AudioTrack
    dialogue: Dialogue
    performance: Optional[Performance] = Field(default=None)
    duration_sec: int = Field(
        ...,
        description="The exact duration of this clip in seconds. Examples: 8, 15, 3, 30, 45.",
    )
    aspect_ratio: str = Field(
        "16:9",
        description="The aspect ratio for this specific clip. Examples: '16:9' (standard widescreen), '9:16' (vertical/mobile), '2.35:1' (cinematic), '4:3' (classic), '1:1' (square).",
    )


class CharacterProfile(BaseModel):
    """A detailed, consistent profile of the character's core attributes."""

    name: str = Field(
        ...,
        description="The primary name of the character. Examples: 'Nyx Cipher', 'Kaelen the Shadowmancer', 'Unit 734', 'Dr. Sarah Chen'.",
    )
    age: int = Field(
        ...,
        description="Character's apparent age. Examples: 27, 350, 5, 72, 16.",
    )
    height: str = Field(
        ...,
        description="Character's height, can include multiple units. Examples: '5'8\" / 173 cm', '7'2\" / 218 cm', '4'11\" / 150 cm', '6'0\" / 183 cm'.",
    )
    build: str = Field(
        ...,
        description="Describes the character's body type and physique. Examples: 'lean, athletic, swimmer's shoulders', 'stocky and muscular', 'delicate and ethereal', 'tall and lanky with dancer's grace', 'compact and powerful'.",
    )
    skin_tone: str = Field(
        ...,
        description="Defines the color and texture of the character's skin. Examples: 'deep bronze with a subtle sun-kissed glow', 'pale porcelain with a dusting of freckles', 'rich ebony with natural luminescence', 'olive-toned with weathered texture', 'metallic, iridescent scales'.",
    )
    hair: str = Field(
        ...,
        description="Describes hair color, length, and style. Examples: 'jet-black, shoulder-length, slicked straight back and dripping', 'silver-white pixie cut with asymmetrical bangs', 'auburn curls cascading past the shoulders', 'buzz-cut platinum blonde', 'bald with intricate henna patterns'.",
    )
    eyes: str = Field(
        ...,
        description="Details the shape and color of the character's eyes. Examples: 'almond-shaped hazel with faint gold flecks', 'wide, ice-blue and piercing', 'deep brown with warm amber highlights', 'green eyes with heterochromia (one blue)', 'glowing crimson without pupils'.",
    )
    distinguishing_marks: str = Field(
        None,
        description="Unique features like tattoos, scars, or piercings. Examples: 'tiny star tattoo tucked behind her right ear; gold stud in upper left helix', 'jagged lightning-bolt scar across the left temple', 'intricate sleeve tattoo depicting ocean waves', 'network of glowing cybernetic implants along the jawline'.",
    )
    demeanour: str = Field(
        ...,
        description="The character's typical personality, mood, and expression. Examples: 'playfully self-assured, almost dare-you smirk', 'stoic and world-weary with gentle eyes', 'manic energy with unpredictable mood swings', 'calm and collected with hidden intensity', 'warm and approachable with infectious laughter'.",
    )
    # -- Fields from former GlobalStyle --
    default_outfit: str = Field(
        ...,
        description="The character's default or primary outfit. Examples: 'metallic-coral bikini, mirrored sunglasses, gold hoop earrings', 'charcoal wool coat over vintage band t-shirt and distressed jeans', 'flowing white linen dress with delicate embroidery', 'tactical black jumpsuit with utility harness', 'three-piece pinstripe suit with pocket watch'.",
    )
    mouth_shape_intensity: float = Field(
        ...,
        description="Controls the exaggeration of mouth movements for lip-syncing (0=subtle, 1=exaggerated). Examples: 0.85, 0.5, 1.0, 0.25.",
    )
    eye_contact_ratio: float = Field(
        ...,
        description="The percentage of time the character should be looking directly at the camera. Examples: 0.7, 0.2, 0.9, 0.5.",
    )


class VideoSchema(BaseModel):
    """The root model, containing a list of characters to be generated."""

    characters: List[CharacterProfile] = Field(
        ...,
        description="A detailed, consistent profile of the character's core attributes.",
    )
    clips: List[Clip] = Field(
        ...,
        description="An array containing definitions for each individual video segment or shot.",
    )


def generate_scenes(
    idea: str,
    output_dir: str,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    if not OPENROUTER_API_KEY:
        raise ValueError("Please set OPENROUTER_API_KEY environment variable")

    # Sanitize the idea to prevent API issues with special characters
    sanitized_idea = sanitize_prompt(idea)
    print(f"Original idea: {idea}")
    print(f"Sanitized idea: {sanitized_idea}")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    # Create a simpler, more focused prompt to avoid token limits
    prompt = f"""Create a JSON video script for this idea: {sanitized_idea}

Requirements:
- Maximum 8 seconds duration
- Include dialogue
- Follow this structure:

{{
  "characters": [{{
    "name": "Character Name",
    "age": 30,
    "height": "6'0\" / 183 cm", 
    "build": "athletic",
    "skin_tone": "description",
    "hair": "description",
    "eyes": "description",
    "distinguishing_marks": "description or null",
    "demeanour": "personality description",
    "default_outfit": "outfit description",
    "mouth_shape_intensity": 0.7,
    "eye_contact_ratio": 0.6
  }}],
  "clips": [{{
    "id": "unique_id",
    "shot": {{
      "composition": "camera shot description",
      "camera_motion": "movement description",
      "frame_rate": "24 fps",
      "film_grain": 0.1,
      "camera": "equipment description"
    }},
    "subject": {{
      "description": "character appearance in scene",
      "wardrobe": "outfit for this scene"
    }},
    "scene": {{
      "location": "where scene takes place",
      "time_of_day": "mid-day",
      "environment": "environmental details"
    }},
    "visual_details": {{
      "action": "what character does",
      "props": "objects in scene or null"
    }},
    "cinematography": {{
      "lighting": "lighting description",
      "tone": "mood/feeling",
      "color_grade": "color scheme"
    }},
    "audio_track": {{
      "lyrics": "dialogue or null",
      "emotion": "vocal emotion or null",
      "flow": "delivery style or null",
      "format": "wav",
      "sample_rate_hz": 48000,
      "channels": 2,
      "style": "music style or null"
    }},
    "dialogue": {{
      "character": "speaking character",
      "line": "spoken text",
      "subtitles": false
    }},
    "duration_sec": 8,
    "aspect_ratio": "16:9"
  }}]
}}

Return ONLY valid JSON, no markdown or extra text."""

    data = {
        "model": "google/gemini-2.5-pro",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 8000,  # Increased token limit
    }

    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1} of {max_retries}...")
            
            response = requests.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=data,
                timeout=60  # Add timeout
            )

            if response.status_code == 429:  # Rate limited
                print(f"Rate limited, waiting {retry_delay * (attempt + 1)} seconds...")
                time.sleep(retry_delay * (attempt + 1))
                continue
                
            if response.status_code != 200:
                print(f"API error {response.status_code}: {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Check if content is empty or just ellipsis
            if not content or content.strip() in ["", "...", "…"]:
                print(f"Empty response on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise Exception("Received empty response from OpenRouter")
            
            # Clean up the content if it has markdown formatting
            if content.startswith("```json"):
                content = content.strip("```json").strip("```")
            elif content.startswith("```"):
                content = content.strip("```")
            
            # Additional cleanup for any remaining markdown
            content = content.strip().strip("`")
            
            # Sanitize the content but don't escape quotes (it's already JSON)
            content = sanitize_prompt(content, escape_for_json=False)
            
            # Check if JSON looks complete (should end with closing brace)
            if not content.strip().endswith('}'):
                print(f"Truncated JSON detected on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise Exception("Received truncated JSON response from OpenRouter")
            
            try:
                video_schema = json.loads(content)
                print(f"Successfully parsed JSON on attempt {attempt + 1}")
                return video_schema
            except json.JSONDecodeError as e:
                print(f"JSON decode error on attempt {attempt + 1}: {e}")
                print(f"Content that failed to parse: {content[:500]}...")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise Exception(f"Failed to parse JSON response from OpenRouter: {e}")
                
        except requests.exceptions.Timeout:
            print(f"Request timeout on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise Exception("Request to OpenRouter timed out")
        except requests.exceptions.RequestException as e:
            print(f"Request error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise Exception(f"Request to OpenRouter failed: {e}")
    
    raise Exception(f"Failed to get valid response after {max_retries} attempts")


def generate_video(
    prompt: str,
    output_dir: str = "videos",
    fname: str = "video.mp4",
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    # Check for Replicate API token (either REPLICATE_API_TOKEN or REPLICATE_API_KEY)
    if not os.getenv("REPLICATE_API_TOKEN") and not os.getenv("REPLICATE_API_KEY"):
        raise ValueError("Please set REPLICATE_API_KEY or REPLICATE_API_TOKEN environment variable")

    # Parse the JSON script to create a more concise prompt for Veo 3
    try:
        script_data = json.loads(prompt)
        
        # Extract key information from the first clip to create a focused prompt
        if script_data.get("clips") and len(script_data["clips"]) > 0:
            clip = script_data["clips"][0]
            
            # Build a concise, descriptive prompt from the script
            video_prompt = f"{clip['cinematography']['tone']} scene: "
            video_prompt += f"{clip['subject']['description']}. "
            video_prompt += f"{clip['visual_details']['action']} "
            video_prompt += f"Location: {clip['scene']['location']}. "
            video_prompt += f"Shot: {clip['shot']['composition']}. "
            video_prompt += f"Lighting: {clip['cinematography']['lighting']}. "
            video_prompt += f"Color grade: {clip['cinematography']['color_grade']}."
            
            if clip.get('dialogue'):
                video_prompt += f" Character says: '{clip['dialogue']['line']}'"
        else:
            # Fallback to using the raw prompt if parsing fails
            video_prompt = prompt[:500]  # Limit length for Veo 3
    except Exception as e:
        print(f"Error parsing JSON script: {e}")
        # If JSON parsing fails, use a truncated version of the prompt
        video_prompt = prompt[:500]

    # Sanitize the video prompt for Veo 3 (don't escape quotes, it's not JSON)
    video_prompt = sanitize_prompt(video_prompt, escape_for_json=False)
    
    # Ensure the prompt isn't too long for Veo 3 (limit to ~400 characters to be safe)
    if len(video_prompt) > 400:
        video_prompt = video_prompt[:397] + "..."

    # Generate video using Replicate
    print(f"Generating video from prompt: {video_prompt[:100]}...")
    print("This may take 30-60 seconds...")
    
    try:
        input_data = {
            "prompt": video_prompt
        }
        
        print("Sending request to Veo 3 Fast...")
        output = replicate.run(
            "google/veo-3-fast",
            input=input_data
        )
        
        # Download and save the video
        video_path = os.path.join(output_dir, fname)
        print(f"Video generation complete! Downloading to {video_path}...")
        
        with open(video_path, "wb") as file:
            file.write(output.read())
        
        print(f"✅ Video saved successfully: {video_path}")
        return video_path
        
    except Exception as e:
        print(f"❌ Error generating video: {e}")
        raise


def generate(
    idea: str,
    output_dir: str = "videos",
    filename: str = "video.mp4",
) -> None:
    """Generates a complete video with multiple scenes."""
    os.makedirs(output_dir, exist_ok=True)

    script = generate_scenes(
        idea=idea,
        output_dir=output_dir,
    )

    generate_video(
        json.dumps(script),
        fname=filename,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    ideas = [
        "Type II civilizations large-scale quantum teleporter with city around it. Dramatic music. No text."
        "Cinematic shot of a dimly lit garage. The Lamborghini Aventadors headlights flicker on. Engine roars to life. Doors rise like wings. Light dances on the sharp curves. No text."
        "Cinematic shot of a sunlit diner counter. A chilled Coca-Cola bottle hisses open, pours itself into a glass. Fizz rises. Light catches the red label. No text."
        "Cinematic shot of a sunlit Scandinavian bedroom. A sealed IKEA box trembles, opens, and flat pack furniture assembles rapidly into a serene, styled room highlighted by a yellow IKEA throw on the bed. No text."
        "A yeti being a confused tourist in central London. No text."
    ]

    for i, idea in enumerate(ideas):
        try:
            print(f"\n{'='*60}")
            print(f"Processing idea {i+1}/{len(ideas)}")
            print(f"{'='*60}")
            
            generate(
                idea=idea,
                filename=f"{idea[:30].lower().replace(' ', '-')}.mp4",
            )
            
            # Add delay between generations to avoid rate limits
            if i < len(ideas) - 1:  # Don't wait after the last one
                print("Waiting 5 seconds before next generation...")
                time.sleep(5)
                
        except Exception as e:
            print(f"Failed to generate video for idea: {idea[:50]}...")
            print(f"Error: {e}")
            continue  # Skip to next idea instead of rerunning
