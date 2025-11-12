# Sentimental Analyzer

A simple web application for **sentiment analysis** — classifying text as **Positive**, **Negative** — built with **Python** and **Streamlit**.

🌐 **Live Demo:** [sentimental-analyzer-pabloma.streamlit.app](https://sentimental-analyzer-pabloma.streamlit.app/)

---

## 🧠 Overview

**Sentimental Analyzer** processes any user-provided text using a pre-trained machine learning model to determine its emotional tone.  
It’s designed for quick and intuitive sentiment evaluation of reviews, comments, or social media posts.

---

## 🗂️ Repository Structure

| File | Description |
|------|--------------|
| `app.py` | Streamlit interface that loads the model and vectorizer, receives input, and displays predictions. |
| `main.py` | Script used to train and prepare the sentiment model and vectorizer. |
| `requirements.txt` | Python dependencies required to run the project. |
| `best_model.pkl` | Pre-trained sentiment classification model (serialized). |
| `vectorizer.pkl` | Text vectorizer used to transform raw text into numerical features. |

---
