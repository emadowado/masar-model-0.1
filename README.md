
gitattributes

*.7z filter=lfs diff=lfs merge=lfs -text
*.arrow filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.bz2 filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text
*.ftz filter=lfs diff=lfs merge=lfs -text
*.gz filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
.lfs. filter=lfs diff=lfs merge=lfs -text
*.mlmodel filter=lfs diff=lfs merge=lfs -text
*.model filter=lfs diff=lfs merge=lfs -text
*.msgpack filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.ot filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.pb filter=lfs diff=lfs merge=lfs -text
*.pickle filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.rar filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
saved_model//* filter=lfs diff=lfs merge=lfs -text
.tar. filter=lfs diff=lfs merge=lfs -text
*.tar filter=lfs diff=lfs merge=lfs -text
*.tflite filter=lfs diff=lfs merge=lfs -text
*.tgz filter=lfs diff=lfs merge=lfs -text
*.wasm filter=lfs diff=lfs merge=lfs -text
*.xz filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
*.zst filter=lfs diff=lfs merge=lfs -text
tfevents filter=lfs diff=lfs merge=lfs -text
saving_info
import csv
import os
DB_FILE = "fan_opinions_log.csv"
def initialize_db():
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["input_text", "output_analysis", "is_correct", "correction"])
def save_opinion(input_text, output_analysis, is_correct, correction):
    with open(DB_FILE, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([input_text, output_analysis, is_correct, correction])
from db import create_opinions_table, insert_opinion
def initialize_db():
    create_opinions_table()
def save_opinion(input_text, output_analysis, is_correct, correction):
    insert_opinion(input_text, output_analysis, is_correct, correction)
requirements.txt
transformers
torch

app.py
import gradio as gr
from transformers import pipeline
import re
import csv
import os
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
LOG_FILE = "fan_opinions_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["input_text", "output_analysis", "is_correct", "correction"])
def extract_info(text, pattern):
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(2).strip() if match else "-"
def analyze_opinion(opinion_text):
    result = sentiment_analyzer(opinion_text)[0]
    label = result["label"]
    score = round(result["score"] * 100, 2)
    sentiment = "🟣 رأي محايد"
    if label == "POSITIVE":
        sentiment = f"🟣 رأي إيجابي ({score}%)"
    elif label == "NEGATIVE":
        sentiment = f"🟣 رأي سلبي ({score}%)"
    team = extract_info(opinion_text, r"(support(?:ing)?|cheering for|I cheer for)\s+([A-Za-z\s]+)")
    name = extract_info(opinion_text, r"(My name is|اسمي)\s+([A-Za-z\s]+)")
    age = extract_info(opinion_text, r"(?:I am|عمري)\s+(\d{1,2})\s+years? old|عمري\s+(\d{1,2})")
    nationality = extract_info(opinion_text, r"(?:I am|أنا)\s+(Saudi|French|Egyptian|Indian|American|British|[A-Za-z]+)")
    stadium = extract_info(opinion_text, r"(stadium|ملعب)\s+([A-Za-z\s]+)")
    day = extract_info(opinion_text, r"(on|في)\s+([A-Za-z]+|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)")
    result_text = f"""🟣 التصنيف: {sentiment}
🟣 الفريق المشجَّع: {team}
🟣 الاسم: {name}
🟣 العمر: {age}
🟣 الجنسية: {nationality}
🟣 الملعب: {stadium}
🟣 اليوم: {day}"""
    return result_text
def full_pipeline(opinion_text, is_correct, correction):
    output = analyze_opinion(opinion_text)
    with open(LOG_FILE, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([opinion_text, output, is_correct, correction])
    return output
custom_css = """
body {
    background-color: #5e17eb;
    color: white;
    font-weight: bold;
    font-size: 18px;
}
textarea, input, select {
    background-color: #4b0dc7 !important;
    color: white !important;
    border: 2px solid white !important;
    font-weight: bold !important;
    font-size: 18px !important;
}
textarea::placeholder {
    color: #ddd !important;
}
.gradio-container {
    font-family: 'Segoe UI', sans-serif;
}
"""
demo = gr.Interface(
    fn=full_pipeline,
    inputs=[
        gr.Textbox(lines=6, label="رأي المشجع"),
        gr.Radio(choices=["نعم", "لا"], label="هل التحليل صحيح؟"),
        gr.Textbox(lines=2, label="إذا لا، ما هو التصحيح؟ (اختياري)")
    ],
    outputs="text",
    title="تحليل آراء المشجعين",
    description="أدخل رأي المشجع، وسنقوم بتحليله. إذا كان التحليل غير دقيق، صححه لنا!",
    theme="default",
    css=custom_css
)
demo.launch()
README.md
metadata
title: Op Masar Ai
emoji: 🚀
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 5.23.3
app_file: app.py
pinned: false
license: apache-2.0


DB!

import sqlite3
DB_NAME = "fan_opinions.db"
def connect_db():
    return sqlite3.connect(DB_NAME)
def create_opinions_table():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS opinions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT NOT NULL,
            output_analysis TEXT NOT NULL,
            is_correct TEXT,
            correction TEXT
        )
    """)
    conn.commit()
    conn.close()
def insert_opinion(input_text, output_analysis, is_correct, correction):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO opinions (input_text, output_analysis, is_correct, correction)
        VALUES (?, ?, ?, ?)
    """, (input_text, output_analysis, is_correct, correction))
    conn.commit()
    conn.close()
