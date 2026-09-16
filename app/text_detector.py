import os
import re
import subprocess
import tempfile

from transformers import pipeline


# =========================================================
# MODELS
# =========================================================

ASR_MODEL_NAME = "openai/whisper-tiny"
TEXT_MODEL_NAME = "rasbt/ai-text-detector-modernbert"


# =========================================================
# LOAD MODELS
# =========================================================

print("Loading speech recognition model...")

speech_recognizer = pipeline(
    "automatic-speech-recognition",
    model=ASR_MODEL_NAME,
    device=-1
)

print("Speech recognition model loaded!")


print("Loading AI text detector...")

text_classifier = pipeline(
    "text-classification",
    model=TEXT_MODEL_NAME,
    device=-1
)

print("AI text detector loaded!")


# =========================================================
# EXTRACT AUDIO FROM VIDEO
# =========================================================

def extract_audio(video_path):

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )

    audio_path = temp_file.name
    temp_file.close()

    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        audio_path
    ]

    try:

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:

            print("FFmpeg audio extraction failed.")

            if os.path.exists(audio_path):
                os.remove(audio_path)

            return None

        return audio_path

    except Exception as error:

        print("Audio extraction error:", error)

        if os.path.exists(audio_path):
            os.remove(audio_path)

        return None


# =========================================================
# AI TEXT SCORE
# =========================================================

def get_text_score(text):

    if not text or not text.strip():
        return 0.0

    try:

        results = text_classifier(
            text[:4000]
        )

        ai_score = 0.0
        human_score = 0.0

        for item in results:

            label = str(
                item.get("label", "")
            ).lower()

            score = float(
                item.get("score", 0.0)
            )

            if (
                "ai" in label
                or "generated" in label
                or "machine" in label
            ):

                ai_score = max(
                    ai_score,
                    score
                )

            elif (
                "human" in label
                or "real" in label
            ):

                human_score = max(
                    human_score,
                    score
                )

            elif label == "label_1":

                ai_score = max(
                    ai_score,
                    score
                )

            elif label == "label_0":

                human_score = max(
                    human_score,
                    score
                )

        total = ai_score + human_score

        if total > 0:

            ai_score /= total

        return float(ai_score)

    except Exception as error:

        print("Text detection error:", error)

        return 0.0


# =========================================================
# TEXT DETECTION
# =========================================================

def detect_text(video_path):

    audio_path = extract_audio(
        video_path
    )

    if audio_path is None:

        return {
            "prediction": "UNAVAILABLE",
            "confidence": 0.0,
            "real_score": 0.0,
            "fake_score": 0.0,
            "language": "Unknown",
            "transcript": "",
            "suspicious_segments": [],
            "available": False
        }

    try:

        # Whisper now receives the extracted WAV file
        result = speech_recognizer(
            audio_path,
            return_timestamps=True
        )

        text = result.get(
            "text",
            ""
        ).strip()

        chunks = result.get(
            "chunks",
            []
        )

        # =================================================
        # NO SPEECH
        # =================================================

        if not text:

            return {
                "prediction": "NO SPEECH",
                "confidence": 0.0,
                "real_score": 1.0,
                "fake_score": 0.0,
                "language": "Unknown",
                "transcript": "",
                "suspicious_segments": [],
                "available": False
            }

        # =================================================
        # OVERALL TEXT ANALYSIS
        # =================================================

        fake_score = get_text_score(
            text
        )

        real_score = 1.0 - fake_score

        if fake_score >= 0.50:

            prediction = "AI-GENERATED"
            confidence = fake_score

        else:

            prediction = "HUMAN"
            confidence = real_score

        # =================================================
        # SUSPICIOUS TEXT TIMESTAMPS
        # =================================================

        suspicious_segments = []

        for chunk in chunks:

            chunk_text = chunk.get(
                "text",
                ""
            ).strip()

            timestamp = chunk.get(
                "timestamp"
            )

            if (
                not chunk_text
                or not timestamp
                or len(timestamp) < 2
            ):
                continue

            start = timestamp[0]
            end = timestamp[1]

            if start is None or end is None:
                continue

            chunk_fake_score = get_text_score(
                chunk_text
            )

            if chunk_fake_score >= 0.50:

                suspicious_segments.append(
                    {
                        "start": float(start),
                        "end": float(end),
                        "fake_score": float(
                            chunk_fake_score
                        ),
                        "text": chunk_text
                    }
                )

        # =================================================
        # LANGUAGE
        # =================================================

        language = detect_language(
            text
        )

        # =================================================
        # RETURN
        # =================================================

        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "real_score": float(real_score),
            "fake_score": float(fake_score),
            "language": language,
            "transcript": text[:1000],
            "suspicious_segments": suspicious_segments,
            "available": True
        }

    except Exception as error:

        print(
            "Speech recognition error:",
            error
        )

        return {
            "prediction": "UNAVAILABLE",
            "confidence": 0.0,
            "real_score": 0.0,
            "fake_score": 0.0,
            "language": "Unknown",
            "transcript": "",
            "suspicious_segments": [],
            "available": False
        }

    finally:

        if os.path.exists(audio_path):

            try:
                os.remove(audio_path)

            except Exception:
                pass


# =========================================================
# LANGUAGE DETECTION
# =========================================================

def detect_language(text):

    tamil_chars = len(
        re.findall(
            r"[\u0B80-\u0BFF]",
            text
        )
    )

    hindi_chars = len(
        re.findall(
            r"[\u0900-\u097F]",
            text
        )
    )

    english_chars = len(
        re.findall(
            r"[A-Za-z]",
            text
        )
    )

    if tamil_chars > english_chars:
        return "Tamil"

    if hindi_chars > english_chars:
        return "Hindi"

    if english_chars > 0:
        return "English"

    return "Unknown"