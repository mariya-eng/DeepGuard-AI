from flask import (
    Flask,
    render_template,
    request,
    redirect,
    session,
    jsonify
)

import os
import uuid
import json

import firebase_admin
from firebase_admin import (
    credentials,
    auth
)

from app.video_detector import detect_video


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "deepguard_ai_secret_key_2026"
)


# =========================================================
# FIREBASE
# =========================================================

SERVICE_ACCOUNT_FILE = "serviceAccountKey.json"

FIREBASE_SERVICE_ACCOUNT_JSON = os.environ.get(
    "FIREBASE_SERVICE_ACCOUNT_JSON"
)

if not firebase_admin._apps:

    if FIREBASE_SERVICE_ACCOUNT_JSON:

        try:

            firebase_config = json.loads(
                FIREBASE_SERVICE_ACCOUNT_JSON
            )

            cred = credentials.Certificate(
                firebase_config
            )

            firebase_admin.initialize_app(
                cred
            )

            print(
                "\nFirebase initialized using "
                "FIREBASE_SERVICE_ACCOUNT_JSON."
            )

        except Exception as e:

            print(
                "\nFirebase environment credential error:"
            )

            print(e)

            raise

    else:

        if not os.path.exists(
            SERVICE_ACCOUNT_FILE
        ):

            raise FileNotFoundError(
                "Firebase credentials not found. "
                "Set FIREBASE_SERVICE_ACCOUNT_JSON "
                "or provide serviceAccountKey.json."
            )

        cred = credentials.Certificate(
            SERVICE_ACCOUNT_FILE
        )

        firebase_admin.initialize_app(
            cred
        )

        print(
            "\nFirebase initialized using "
            "serviceAccountKey.json."
        )


# =========================================================
# UPLOAD
# =========================================================

UPLOAD_FOLDER = "uploads"

ALLOWED_EXTENSIONS = {
    "mp4",
    "avi",
    "mov",
    "mkv"
}

MAX_FILE_SIZE = (
    100 * 1024 * 1024
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config[
    "UPLOAD_FOLDER"
] = UPLOAD_FOLDER

app.config[
    "MAX_CONTENT_LENGTH"
] = MAX_FILE_SIZE


# =========================================================
# HELPERS
# =========================================================

def allowed_file(filename):

    return (
        "."
        in filename
        and filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


def format_duration(seconds):

    try:

        seconds = float(seconds)

    except (
        TypeError,
        ValueError
    ):

        return "Unknown"

    total_seconds = int(
        round(seconds)
    )

    minutes = (
        total_seconds // 60
    )

    remaining_seconds = (
        total_seconds % 60
    )

    if minutes > 0:

        return (
            f"{minutes} minute(s) "
            f"{remaining_seconds} second(s)"
        )

    return (
        f"{remaining_seconds} second(s)"
    )


def get_current_user():

    return session.get(
        "user"
    )


def get_analysis_result():

    return session.get(
        "analysis_result"
    )


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    user = get_current_user()

    # -----------------------------------------------------
    # IMPORTANT:
    # Do not show an old analysis after page refresh.
    # Analysis results are displayed only immediately
    # after a new analysis.
    # -----------------------------------------------------

    result = session.pop(
        "analysis_result",
        None
    )

    session.pop(
        "uploaded_video",
        None
    )

    user_email = None

    if user:

        user_email = user.get(
            "email",
            ""
        )

    return render_template(
        "index.html",
        user_email=user_email,
        result=result
    )


# =========================================================
# FIREBASE LOGIN
# =========================================================

@app.route(
    "/firebase-login",
    methods=["POST"]
)
def firebase_login():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "success": False,
                "message":
                    "Login data not received."
            }), 400

        id_token = data.get(
            "idToken"
        )

        if not id_token:

            return jsonify({
                "success": False,
                "message":
                    "Firebase ID token is missing."
            }), 400

        decoded_token = (
            auth.verify_id_token(
                id_token
            )
        )

        uid = decoded_token.get(
            "uid"
        )

        email = decoded_token.get(
            "email",
            ""
        )

        session.pop(
            "analysis_result",
            None
        )

        session.pop(
            "uploaded_video",
            None
        )

        session["user"] = {
            "uid": uid,
            "email": email
        }

        session.modified = True

        print(
            "\nFIREBASE LOGIN SUCCESS"
        )

        return jsonify({
            "success": True,
            "message":
                "Login successful."
        })

    except Exception as e:

        print(
            "\nFIREBASE LOGIN ERROR:"
        )

        print(e)

        return jsonify({
            "success": False,
            "message":
                "Firebase authentication failed."
        }), 401


# =========================================================
# ANALYZE VIDEO
# =========================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze_video():

    if "user" not in session:

        return """
        <script>
            alert("Please login first.");
            window.location.href = "/";
        </script>
        """

    if "video" not in request.files:

        return """
        <script>
            alert("No video file selected.");
            window.location.href = "/";
        </script>
        """

    video = request.files[
        "video"
    ]

    if video.filename == "":

        return """
        <script>
            alert("Please select a video.");
            window.location.href = "/";
        </script>
        """

    if not allowed_file(
        video.filename
    ):

        return """
        <script>
            alert("Invalid video format.");
            window.location.href = "/";
        </script>
        """

    extension = (
        video.filename
        .rsplit(
            ".",
            1
        )[1]
        .lower()
    )

    filename = (
        f"{uuid.uuid4().hex}"
        f".{extension}"
    )

    video_path = os.path.join(
        app.config[
            "UPLOAD_FOLDER"
        ],
        filename
    )

    try:

        video.save(
            video_path
        )

        print(
            "\n=============================="
        )

        print(
            "MULTIMODAL ANALYSIS STARTED"
        )

        print(
            "=============================="
        )

        print(
            "\nStarting Visual + Audio + Text analysis..."
        )

        # =================================================
        # MULTIMODAL ANALYSIS
        # =================================================

        analysis = detect_video(
            video_path,
            max_frames=20
        )

        # =================================================
        # GET INDIVIDUAL MODALITY RESULTS
        # =================================================

        visual_result = analysis.get(
            "visual_analysis",
            {}
        )

        audio_result = analysis.get(
            "audio_analysis",
            {}
        )

        text_result = analysis.get(
            "text_analysis",
            {}
        )

        # =================================================
        # OVERALL RESULT
        # =================================================

        final_prediction = analysis.get(
            "prediction",
            "UNKNOWN"
        )

        final_confidence = float(
            analysis.get(
                "confidence",
                0.0
            )
        )

        overall_real = float(
            analysis.get(
                "real_score",
                0.0
            )
        )

        overall_fake = float(
            analysis.get(
                "fake_score",
                0.0
            )
        )

        # =================================================
        # FINAL RESULT OBJECT
        # =================================================

        result = {

            # -------------------------------------------------
            # OVERALL
            # -------------------------------------------------

            "prediction":
                final_prediction,

            "confidence":
                final_confidence,

            "real_score":
                overall_real,

            "fake_score":
                overall_fake,

            # -------------------------------------------------
            # VIDEO INFORMATION
            # -------------------------------------------------

            "frames_analyzed":
                analysis.get(
                    "frames_analyzed",
                    0
                ),

            "duration_seconds":
                analysis.get(
                    "duration_seconds",
                    0
                ),

            # -------------------------------------------------
            # VISUAL / IMAGE
            # -------------------------------------------------

            "visual":
                visual_result,

            # -------------------------------------------------
            # AUDIO / VOICE
            # -------------------------------------------------

            "audio":
                audio_result,

            # -------------------------------------------------
            # TEXT / SPEECH
            # -------------------------------------------------

            "text":
                text_result,

            # -------------------------------------------------
            # COMPATIBILITY
            # -------------------------------------------------

            "suspicious_segments":
                analysis.get(
                    "suspicious_segments",
                    []
                )
        }

        # =================================================
        # SAVE RESULT TO SESSION
        # =================================================

        session[
            "analysis_result"
        ] = result

        session[
            "uploaded_video"
        ] = {
            "filename":
                video.filename
        }

        session.modified = True

        # =================================================
        # TERMINAL OUTPUT
        # =================================================

        print(
            "\n=============================="
        )

        print(
            "MULTIMODAL ANALYSIS COMPLETED"
        )

        print(
            "=============================="
        )

        print(
            "Final Prediction:",
            final_prediction
        )

        print(
            "Overall Real:",
            round(
                overall_real * 100,
                2
            ),
            "%"
        )

        print(
            "Overall Fake:",
            round(
                overall_fake * 100,
                2
            ),
            "%"
        )

        print(
            "Visual Prediction:",
            visual_result.get(
                "prediction",
                "UNKNOWN"
            )
        )

        print(
            "Visual Fake:",
            round(
                float(
                    visual_result.get(
                        "fake_score",
                        0
                    )
                ) * 100,
                2
            ),
            "%"
        )

        print(
            "Audio Prediction:",
            audio_result.get(
                "prediction",
                "UNKNOWN"
            )
        )

        print(
            "Audio Fake:",
            round(
                float(
                    audio_result.get(
                        "fake_score",
                        0
                    )
                ) * 100,
                2
            ),
            "%"
        )

        print(
            "Text Prediction:",
            text_result.get(
                "prediction",
                "UNKNOWN"
            )
        )

        print(
            "Text AI Score:",
            round(
                float(
                    text_result.get(
                        "fake_score",
                        0
                    )
                ) * 100,
                2
            ),
            "%"
        )

        print(
            "==============================\n"
        )

        return render_template(
            "index.html",
            user_email=session[
                "user"
            ].get(
                "email",
                ""
            ),
            result=result
        )

    except Exception as e:

        print(
            "\nVIDEO ANALYSIS ERROR:"
        )

        print(e)

        return """
        <script>
            alert(
                "Video analysis failed. Check the terminal."
            );
            window.location.href = "/";
        </script>
        """

    finally:

        if os.path.exists(
            video_path
        ):

            try:

                os.remove(
                    video_path
                )

            except Exception:

                pass


# =========================================================
# CHATBOT
# =========================================================

@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    if "user" not in session:

        return jsonify({
            "answer":
                "Please login first."
        })

    data = request.get_json()

    if not data:

        return jsonify({
            "answer":
                "Please enter a question."
        })

    question = data.get(
        "question",
        ""
    ).strip().lower()

    if not question:

        return jsonify({
            "answer":
                "Please type a question."
        })

    result = get_analysis_result()

    if not result:

        return jsonify({
            "answer":
                "Please upload and analyze a video first."
        })

    prediction = str(
        result.get(
            "prediction",
            "UNKNOWN"
        )
    )

    confidence = float(
        result.get(
            "confidence",
            0
        )
    )

    real_score = float(
        result.get(
            "real_score",
            0
        )
    )

    fake_score = float(
        result.get(
            "fake_score",
            0
        )
    )

    # =====================================================
    # WHERE IS FAKE?
    # =====================================================

    if (
        "where" in question
        and (
            "fake" in question
            or "deepfake" in question
        )
    ):

        visual = result.get(
            "visual",
            {}
        )

        segments = visual.get(
            "suspicious_segments",
            []
        )

        if not segments:

            return jsonify({
                "answer":
                    "No suspicious visual "
                    "area was detected."
            })

        answer = (
            "Fake visual content "
            "was detected at:\n\n"
        )

        for segment in segments:

            start = float(
                segment.get(
                    "start",
                    0
                )
            )

            end = float(
                segment.get(
                    "end",
                    0
                )
            )

            answer += (
                f"• "
                f"{int(start // 60):02d}:"
                f"{int(start % 60):02d}"
                f" – "
                f"{int(end // 60):02d}:"
                f"{int(end % 60):02d}\n"
            )

        return jsonify({
            "answer": answer
        })

    # =====================================================
    # AUDIO
    # =====================================================

    if (
        "audio" in question
        or "voice" in question
    ):

        audio = result.get(
            "audio",
            {}
        )

        return jsonify({
            "answer":
                "AUDIO ANALYSIS\n\n"
                f"Status: "
                f"{audio.get('prediction', 'UNKNOWN')}\n"
                f"Real Score: "
                f"{float(audio.get('real_score', 0)) * 100:.2f}%\n"
                f"Fake Score: "
                f"{float(audio.get('fake_score', 0)) * 100:.2f}%\n"
                f"Confidence: "
                f"{float(audio.get('confidence', 0)) * 100:.2f}%"
        })

    # =====================================================
    # TEXT
    # =====================================================

    if (
        "text" in question
        or "speech" in question
        or "transcript" in question
        or "language" in question
    ):

        text = result.get(
            "text",
            {}
        )

        return jsonify({
            "answer":
                "TEXT / SPEECH ANALYSIS\n\n"
                f"Status: "
                f"{text.get('prediction', 'UNKNOWN')}\n"
                f"Real Score: "
                f"{float(text.get('real_score', 0)) * 100:.2f}%\n"
                f"AI Score: "
                f"{float(text.get('fake_score', 0)) * 100:.2f}%\n"
                f"Language: "
                f"{text.get('language', 'Unknown')}"
        })

    # =====================================================
    # VISUAL / IMAGE
    # =====================================================

    if (
        "visual" in question
        or "image" in question
        or "video" in question
    ):

        visual = result.get(
            "visual",
            {}
        )

        return jsonify({
            "answer":
                "VISUAL / IMAGE ANALYSIS\n\n"
                f"Prediction: "
                f"{visual.get('prediction', 'UNKNOWN')}\n"
                f"Real Score: "
                f"{float(visual.get('real_score', 0)) * 100:.2f}%\n"
                f"Fake Score: "
                f"{float(visual.get('fake_score', 0)) * 100:.2f}%\n"
                f"Confidence: "
                f"{float(visual.get('confidence', 0)) * 100:.2f}%"
        })

    # =====================================================
    # OVERALL
    # =====================================================

    if (
        "real or fake" in question
        or "prediction" in question
        or "result" in question
        or "deepfake" in question
    ):

        return jsonify({
            "answer":
                f"Final Prediction: "
                f"{prediction}\n\n"
                f"Overall Confidence: "
                f"{confidence * 100:.2f}%\n"
                f"Real Score: "
                f"{real_score * 100:.2f}%\n"
                f"Fake Score: "
                f"{fake_score * 100:.2f}%"
        })

    # =====================================================
    # DEFAULT
    # =====================================================

    return jsonify({
        "answer":
            "I can answer questions about:\n\n"
            "• Final prediction\n"
            "• Overall confidence\n"
            "• Visual / image analysis\n"
            "• Audio / voice analysis\n"
            "• Text / speech analysis\n"
            "• Where fake visual content occurs\n"
            "• Detected language"
    })


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/")


# =========================================================
# FILE TOO LARGE
# =========================================================

@app.errorhandler(413)
def file_too_large(error):

    return """
    <script>
        alert(
            "Video file is too large. Maximum size is 100 MB."
        );
        window.location.href = "/";
    </script>
    """, 413


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            7860
        )
    )

    print(
        "\n===================================="
    )

    print(
        "        DEEPGUARD AI SERVER"
    )

    print(
        "===================================="
    )

    print(
        f"http://127.0.0.1:{port}"
    )

    print(
        "====================================\n"
    )

    app.run(
        debug=False,
        host="0.0.0.0",
        port=port
    )