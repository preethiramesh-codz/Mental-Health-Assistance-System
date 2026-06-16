import streamlit as st
import joblib
import pandas as pd
from textblob import TextBlob
from datetime import datetime
import os

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Mental Health Monitoring System",
    page_icon="🧠",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================

model = joblib.load("mental_health_model.pkl")

# =========================
# SIDEBAR
# =========================

st.sidebar.title("🧠 AI Mental Health Dashboard")

st.sidebar.success("✅ Mental State Prediction")
st.sidebar.success("✅ Sentiment Analysis")
st.sidebar.success("✅ Emotion Detection")
st.sidebar.success("✅ Risk Assessment")
st.sidebar.success("✅ Confidence Score")
st.sidebar.success("✅ Wellness Recommendations")
st.sidebar.success("✅ NLP-Based Analysis")
st.sidebar.success("✅ History Monitoring")

# =========================
# TITLE
# =========================

st.title("🧠 AI-Based Mental Health Monitoring System")

st.subheader(
    "Personalized Mental Health Analysis & Wellness Recommendation Platform"
)

st.write("""
This AI system analyzes user emotions, predicts mental health categories,
performs sentiment analysis, evaluates risk level, calculates wellness score,
and provides personalized recommendations.
""")

# =========================
# USER INPUT
# =========================

user_input = st.text_area(
    "💬 Describe how you are feeling",
    height=200
)

# =========================
# ANALYSIS BUTTON
# =========================

if st.button("🔍 Analyze Mental State"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        # =====================
        # PREDICTION
        # =====================

        prediction = model.predict([user_input])[0]

        try:
            confidence = max(
                model.predict_proba([user_input])[0]
            ) * 100
        except:
            confidence = 90

        # =====================
        # SENTIMENT ANALYSIS
        # =====================

        sentiment = TextBlob(
            user_input
        ).sentiment.polarity

        if sentiment > 0:
            sentiment_label = "Positive 😊"
        elif sentiment < 0:
            sentiment_label = "Negative 😔"
        else:
            sentiment_label = "Neutral 😐"

        # =====================
        # EMOTION DETECTION
        # =====================

        text_lower = user_input.lower()

        if any(word in text_lower for word in [
            "sad", "hopeless", "depressed",
            "cry", "alone", "helpless"
        ]):
            emotion = "Sadness 😔"

        elif any(word in text_lower for word in [
            "anxious", "worried",
            "nervous", "panic"
        ]):
            emotion = "Anxiety 😟"

        elif any(word in text_lower for word in [
            "angry",
            "frustrated",
            "annoyed"
        ]):
            emotion = "Anger 😠"

        elif any(word in text_lower for word in [
            "happy",
            "great",
            "excited",
            "joy"
        ]):
            emotion = "Happiness 😊"

        else:
            emotion = "Neutral 😐"

        # =====================
        # SMART RISK ASSESSMENT
        # =====================

        high_risk_words = [
            "suicide",
            "kill myself",
            "end my life",
            "hopeless",
            "worthless",
            "helpless"
        ]

        if any(word in text_lower for word in high_risk_words):

            risk = "High Risk 🔴"
            wellness_score = 25

        elif sentiment < 0:

            risk = "Medium Risk 🟠"
            wellness_score = 50

        else:

            risk = "Low Risk 🟢"
            wellness_score = 85

        # =====================
        # RESULTS
        # =====================

        st.success(
            f"Predicted Mental State: {prediction}"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Mental State",
                prediction
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        with col3:
            st.metric(
                "Wellness",
                f"{wellness_score}/100"
            )

        with col4:
            st.metric(
                "Emotion",
                emotion
            )

        # =====================
        # SENTIMENT
        # =====================

        st.subheader("📊 Sentiment Analysis")
        st.info(sentiment_label)

        # =====================
        # RISK
        # =====================

        st.subheader("⚠ Risk Assessment")
        st.warning(risk)

        if "High Risk" in risk:

            st.error("""
🚨 CRISIS ALERT

The entered text indicates severe emotional distress.

Immediate support from trusted individuals,
mental health professionals, or emergency
support services is strongly recommended.
""")

        # =====================
        # WELLNESS SCORE
        # =====================

        st.subheader("💚 Wellness Score")

        st.progress(wellness_score)

        st.success(
            f"Wellness Score: {wellness_score}/100"
        )

        # =====================
        # ANALYTICS GRAPH
        # =====================

        st.subheader("📊 Analysis Metrics")

        chart_data = pd.DataFrame({
            "Metric": [
                "Confidence",
                "Wellness"
            ],
            "Score": [
                confidence,
                wellness_score
            ]
        })

        st.bar_chart(
            chart_data.set_index("Metric")
        )

        # =====================
        # RECOMMENDATIONS
        # =====================

        st.subheader(
            "🌿 Personalized Recommendations"
        )

        prediction_lower = prediction.lower()

        if prediction_lower == "anxiety":

            st.info("""
• Practice deep breathing exercises

• Try mindfulness meditation

• Reduce caffeine intake

• Maintain healthy sleep patterns

• Take breaks from stressful activities
""")

        elif prediction_lower == "depression":

            st.info("""
• Stay connected with friends and family

• Exercise regularly

• Maintain a daily routine

• Spend time outdoors

• Consider professional support
""")

        elif prediction_lower == "stress":

            st.info("""
• Prioritize important tasks

• Take short relaxation breaks

• Practice yoga or meditation

• Maintain work-life balance

• Get sufficient sleep
""")

        elif prediction_lower == "suicidal":

            st.error("""
🚨 High Risk Detected

Please immediately contact:

• Trusted family member

• Mental health professional

• Emergency support services

You deserve support and help.
""")

        else:

            st.info("""
• Maintain a healthy lifestyle

• Stay socially connected

• Exercise regularly

• Practice self-care

• Monitor emotional wellbeing
""")

        # =====================
        # SAVE HISTORY
        # =====================

        history_file = "history/history.csv"

        new_data = pd.DataFrame({
            "Date": [datetime.now()],
            "Prediction": [prediction],
            "Emotion": [emotion],
            "Sentiment": [sentiment_label],
            "Risk": [risk],
            "WellnessScore": [wellness_score],
            "Input": [user_input]
        })

        try:

            if os.path.exists(history_file):

                old_data = pd.read_csv(history_file)

                updated_data = pd.concat(
                    [old_data, new_data],
                    ignore_index=True
                )

                updated_data.to_csv(
                    history_file,
                    index=False
                )

            else:

                new_data.to_csv(
                    history_file,
                    index=False
                )

        except:
            pass

# =========================
# HISTORY SECTION
# =========================

st.markdown("---")

st.subheader(
    "📊 Mental Health Monitoring History"
)

history_file = "history/history.csv"

if os.path.exists(history_file):

    history = pd.read_csv(history_file)

    st.dataframe(
        history.tail(20),
        use_container_width=True
    )

    if "WellnessScore" in history.columns:

        st.subheader("📈 Wellness Trend")

        st.line_chart(
            history["WellnessScore"]
        )

else:

    st.info(
        "No history available."
    )

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "Developed by Preethi | AI-Based Mental Health Monitoring and Wellness Recommendation System"
)
