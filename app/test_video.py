from app.video_detector import detect_video

result = detect_video("test.mp4", max_frames=20)

print("\n==============================")
print("DEEPFAKE DETECTION RESULT")
print("==============================")

print("Prediction :", result["prediction"])
print("Confidence :", round(result["confidence"] * 100, 2), "%")
print("Real Score :", round(result["real_score"] * 100, 2), "%")
print("Fake Score :", round(result["fake_score"] * 100, 2), "%")
print("Frames     :", result["frames_analyzed"])