import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf
from transformers import pipeline


MODEL_NAME = "Hemgg/Deepfake-audio-detection"

audio_classifier = None


def load_audio_model():
    global audio_classifier

    if audio_classifier is None:
        print("Loading AI audio detector...")
        audio_classifier = pipeline(
            "audio-classification",
            model=MODEL_NAME,
            device=-1
        )
        print("AI audio detector loaded!")

    return audio_classifier


def extract_audio(video_path):
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )
    temp_audio_path = temp_file.name
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
        temp_audio_path
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            return None

        return temp_audio_path

    except Exception:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        return None


def get_label_scores(results):
    fake_score = 0.0
    real_score = 0.0

    for item in results:
        label = str(item.get("label", "")).lower()
        score = float(item.get("score", 0.0))

        if (
            "fake" in label
            or "deepfake" in label
            or "spoof" in label
            or "synthetic" in label
            or "ai" in label
            or "generated" in label
        ):
            fake_score = max(fake_score, score)

        elif (
            "real" in label
            or "human" in label
            or "bonafide" in label
            or "genuine" in label
        ):
            real_score = max(real_score, score)

        elif label == "label_1":
            fake_score = max(fake_score, score)

        elif label == "label_0":
            real_score = max(real_score, score)

    total = fake_score + real_score

    if total > 0:
        fake_score /= total
        real_score /= total

    return real_score, fake_score


def merge_segments(segments):
    if not segments:
        return []

    segments = sorted(
        segments,
        key=lambda x: x["start"]
    )

    merged = [segments[0].copy()]

    for current in segments[1:]:
        previous = merged[-1]

        if current["start"] <= previous["end"] + 0.5:
            previous["end"] = max(
                previous["end"],
                current["end"]
            )
            previous["fake_score"] = max(
                previous["fake_score"],
                current["fake_score"]
            )
        else:
            merged.append(current.copy())

    return merged


def detect_audio(video_path):
    audio_path = extract_audio(video_path)

    if audio_path is None:
        return {
            "available": False,
            "prediction": "NO AUDIO",
            "confidence": 0.0,
            "real_score": 0.0,
            "fake_score": 0.0,
            "chunks_analyzed": 0,
            "suspicious_segments": []
        }

    try:
        classifier = load_audio_model()

        audio, sample_rate = sf.read(
            audio_path,
            dtype="float32"
        )

        if len(audio) == 0:
            return {
                "available": False,
                "prediction": "NO AUDIO",
                "confidence": 0.0,
                "real_score": 0.0,
                "fake_score": 0.0,
                "chunks_analyzed": 0,
                "suspicious_segments": []
            }

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        chunk_duration = 4
        hop_duration = 2

        chunk_size = sample_rate * chunk_duration
        hop_size = sample_rate * hop_duration

        results_all = []
        suspicious_segments = []

        total_samples = len(audio)

        start_sample = 0

        while start_sample < total_samples:
            end_sample = min(
                start_sample + chunk_size,
                total_samples
            )

            chunk = audio[start_sample:end_sample]

            if len(chunk) < sample_rate:
                break

            chunk_results = classifier(
                {
                    "array": chunk,
                    "sampling_rate": sample_rate
                }
            )

            real_score, fake_score = get_label_scores(
                chunk_results
            )

            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate

            results_all.append(
                {
                    "real_score": real_score,
                    "fake_score": fake_score
                }
            )

            if fake_score >= 0.50:
                suspicious_segments.append(
                    {
                        "start": float(start_time),
                        "end": float(end_time),
                        "fake_score": float(fake_score)
                    }
                )

            start_sample += hop_size

        if not results_all:
            return {
                "available": False,
                "prediction": "NO AUDIO",
                "confidence": 0.0,
                "real_score": 0.0,
                "fake_score": 0.0,
                "chunks_analyzed": 0,
                "suspicious_segments": []
            }

        average_real = float(
            np.mean(
                [x["real_score"] for x in results_all]
            )
        )

        average_fake = float(
            np.mean(
                [x["fake_score"] for x in results_all]
            )
        )

        if average_fake >= 0.50:
            prediction = "FAKE"
            confidence = average_fake
        else:
            prediction = "REAL"
            confidence = average_real

        merged_segments = merge_segments(
            suspicious_segments
        )

        return {
            "available": True,
            "prediction": prediction,
            "confidence": float(confidence),
            "real_score": float(average_real),
            "fake_score": float(average_fake),
            "chunks_analyzed": len(results_all),
            "suspicious_segments": merged_segments
        }

    except Exception as error:
        print("Audio detection error:", error)

        return {
            "available": False,
            "prediction": "UNAVAILABLE",
            "confidence": 0.0,
            "real_score": 0.0,
            "fake_score": 0.0,
            "chunks_analyzed": 0,
            "suspicious_segments": []
        }

    finally:
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass