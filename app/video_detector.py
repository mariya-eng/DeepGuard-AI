import cv2
import numpy as np

from app.model import predict_frame
from app.audio_detector import detect_audio
from app.text_detector import detect_text


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

        if current["start"] <= previous["end"] + 1.0:
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


def detect_video(video_path, max_frames=20):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0:
        fps = 30.0

    duration_seconds = (
        total_frames / fps
        if total_frames > 0
        else 0.0
    )

    frame_predictions = []

    if total_frames > 0:
        frame_indexes = np.linspace(
            0,
            total_frames - 1,
            min(max_frames, total_frames),
            dtype=int
        )
    else:
        frame_indexes = []

    for frame_index in frame_indexes:
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index)
        )

        success, frame = cap.read()

        if not success:
            continue

        timestamp = (
            float(frame_index) / fps
        )

        try:
            results = predict_frame(frame)

            fake_score = 0.0
            real_score = 0.0

            for item in results:
                label = str(
                    item.get("label", "")
                ).lower()

                score = float(
                    item.get("score", 0.0)
                )

                if (
                    "fake" in label
                    or "deepfake" in label
                    or "ai" in label
                    or "generated" in label
                ):
                    fake_score = max(
                        fake_score,
                        score
                    )

                elif (
                    "real" in label
                    or "original" in label
                    or "genuine" in label
                ):
                    real_score = max(
                        real_score,
                        score
                    )

                elif label == "label_1":
                    fake_score = max(
                        fake_score,
                        score
                    )

                elif label == "label_0":
                    real_score = max(
                        real_score,
                        score
                    )

            total = fake_score + real_score

            if total > 0:
                fake_score /= total
                real_score /= total

            frame_predictions.append(
                {
                    "timestamp": timestamp,
                    "fake_score": fake_score,
                    "real_score": real_score
                }
            )

        except Exception as error:
            print(
                f"Frame analysis error at "
                f"{timestamp:.2f}s:",
                error
            )

    cap.release()

    if not frame_predictions:
        raise ValueError(
            "No video frames could be analyzed."
        )

    # -------------------------------------------------
    # VISUAL / IMAGE ANALYSIS
    # -------------------------------------------------

    average_fake = float(
        np.mean(
            [
                item["fake_score"]
                for item in frame_predictions
            ]
        )
    )

    average_real = float(
        np.mean(
            [
                item["real_score"]
                for item in frame_predictions
            ]
        )
    )

    if average_fake >= 0.50:
        prediction = "DEEPFAKE"
        confidence = average_fake
    else:
        prediction = "ORIGINAL"
        confidence = average_real

    # -------------------------------------------------
    # VISUAL SUSPICIOUS TIMESTAMPS
    # -------------------------------------------------

    suspicious_segments = []

    for index, item in enumerate(frame_predictions):

        if item["fake_score"] < 0.50:
            continue

        start_time = item["timestamp"]

        if index + 1 < len(frame_predictions):
            end_time = (
                frame_predictions[index + 1]["timestamp"]
            )
        else:
            end_time = duration_seconds

        if end_time <= start_time:
            end_time = start_time + 1.0

        suspicious_segments.append(
            {
                "start": float(start_time),
                "end": float(end_time),
                "fake_score": float(
                    item["fake_score"]
                )
            }
        )

    merged_visual_segments = merge_segments(
        suspicious_segments
    )

    # -------------------------------------------------
    # AUDIO / VOICE ANALYSIS
    # -------------------------------------------------

    try:
        audio_analysis = detect_audio(
            video_path
        )

    except Exception as error:
        print(
            "Audio analysis error:",
            error
        )

        audio_analysis = {
            "available": False,
            "prediction": "UNAVAILABLE",
            "confidence": 0.0,
            "real_score": 0.0,
            "fake_score": 0.0,
            "chunks_analyzed": 0,
            "suspicious_segments": []
        }

    # -------------------------------------------------
    # SPEECH / TEXT ANALYSIS
    # -------------------------------------------------

    try:
        text_analysis = detect_text(
            video_path
        )

    except Exception as error:
        print(
            "Text analysis error:",
            error
        )

        text_analysis = {
            "available": False,
            "prediction": "UNAVAILABLE",
            "confidence": 0.0,
            "real_score": 0.0,
            "fake_score": 0.0,
            "language": "Unknown",
            "transcript": "",
            "suspicious_segments": []
        }

    # -------------------------------------------------
    # FINAL RESULT
    # -------------------------------------------------

    return {
        # Existing overall visual result
        "prediction": prediction,
        "confidence": float(confidence),
        "fake_score": float(average_fake),
        "real_score": float(average_real),
        "frames_analyzed": len(
            frame_predictions
        ),
        "duration_seconds": float(
            duration_seconds
        ),

        # Existing visual timestamps
        "suspicious_segments": (
            merged_visual_segments
        ),

        # Detailed visual/image analysis
        "visual_analysis": {
            "prediction": prediction,
            "confidence": float(confidence),
            "fake_score": float(average_fake),
            "real_score": float(average_real),
            "frames_analyzed": len(
                frame_predictions
            ),
            "suspicious_segments": (
                merged_visual_segments
            )
        },

        # Audio/voice analysis
        "audio_analysis": audio_analysis,

        # Speech/text analysis
        "text_analysis": text_analysis
    }