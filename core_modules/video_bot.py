import glob
import re
import faiss
import numpy as np
import sys
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All
from TTS.api import TTS
import subprocess
import time
import os
from moviepy.editor import VideoFileClip
import wave

os.environ['CURL_CA_BUNDLE'] = ''

speaker_wav = "Ratan_Tataa.wav"
IMG_PATH = "final_rt_image.jpg"
RESULT_DIR = "results"
AUDIO_PATH = "temp_ratan_tata.wav"

# 1. Load transcript files
knowledge_texts = []
for file in glob.glob('transcripts/*.txt'):
    with open(file, 'r', encoding='utf-8') as f:
        knowledge_texts.append(f.read())
if not knowledge_texts:
    raise ValueError("No .txt files found in 'transcripts' folder.")

# 2. Chunking
def chunk_text(text, max_tokens=150):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current, token_count = [], [], 0
    for sent in sentences:
        token_count += len(sent.split())
        current.append(sent)
        if token_count >= max_tokens:
            chunks.append(' '.join(current))
            current, token_count = [], 0
    if current:
        chunks.append(' '.join(current))
    return chunks

all_chunks = []
for doc in knowledge_texts:
    all_chunks.extend(chunk_text(doc))
if not all_chunks:
    raise ValueError("No text chunks found in input files.")

# 3. Embeddings & FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = model.encode(all_chunks)
np_emb = np.array(chunk_embeddings).astype('float32')
index = faiss.IndexFlatL2(np_emb.shape[1])
index.add(np_emb)

# 4. Load GPT4All model & TTS
model_path = "C:\\Users\\hp\\.gpt4all\\models\\mistral-7b-instruct-v0.2.Q4_0.gguf"
llm = GPT4All(model_path, allow_download=False)
tts = TTS(model_name="tts_models/multilingual/xtts_v2")

def get_bot_answer(user_query):
    query_emb = model.encode([user_query]).astype('float32')
    k = 3
    D, I = index.search(query_emb, k)
    context = "\n".join([all_chunks[idx] for idx in I[0]])
    prompt = (
        "Answer as Ratan Tata's expert. Use only the context below. Reply in ONE complete sentence (around 20 to 25 words). No lists. No disclaimers.\n"
        f"Context:\n{context}\n"
        f"Question: {user_query}\n"
        f"Answer:"
    )
    with llm.chat_session():
        response = llm.generate(prompt, max_tokens=20)
    return response.strip()

def get_audio_duration(path):
    with wave.open(path, 'rb') as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)

print("\n=== Ratan Tata Video Knowledge Bot ===\n\nPlease Include Question mark at the end of the question(?)\n")
print("\nAsk anything about Ratan Tata. Type 'exit' to quit.\n")

while True:
    user_query = input("Ask a question: ").strip()
    if user_query.lower() == 'exit':
        print("Goodbye!")
        break

    if user_query.lower() in [
        "who are you", "what are you", "what is your name", "who is this",
        "who are you?", "what are you?", "what is your name?", "who is this?"
    ]:
        print("Hello! I am the Ratan Tata Knowledge Bot—your expert assistant here to answer questions about Ratan Tata, his legacy, achievements, and work. Feel free to ask me anything related to Ratan Tata, and I'll do my best to help you.")
        continue

    if user_query.lower() in [
        "thanks", "thank you", "thank you so much",
        "thank you so much for your info about ratan tata",
        "thank you so much for the info"
    ]:
        print("You're welcome! If you have any more questions about Ratan Tata or need further information, just ask.")
        continue

    if not ("ratan tata" in user_query.lower() or "tata group" in user_query.lower()
            or "he" in user_query.lower() or "him" in user_query.lower()
            or "tata sons" in user_query.lower() or "tata" in user_query.lower()):
        print("This is not relevant to Ratan Tata. Please ask about Ratan Tata!")
        continue

    start_time = time.time()
    answer = get_bot_answer(user_query)
    print("Bot (text):", answer)
    print("Answer length (words):", len(answer.split()))

    tts.tts_to_file(
        text=answer,
        speaker_wav=speaker_wav,
        language="en",
        file_path=AUDIO_PATH
    )

    duration = get_audio_duration(AUDIO_PATH)
    print("Audio duration (seconds):", round(duration, 2))
    if duration > 15:
        print("Audio too long for fast video synthesis. Please ask a shorter question or tune LLM settings.")

    print("Bot (video): Generating animated lipsynced video...")
    subprocess.run([
        sys.executable, "inference.py",
        "--driven_audio", AUDIO_PATH,
        "--source_image", IMG_PATH,
        "--result_dir", RESULT_DIR
        # Uncomment these for higher quality but slower output:
        # "--enhancer", "gfpgan"
    ])

    result_videos = glob.glob(os.path.join(RESULT_DIR, "**", "*.mp4"), recursive=True)
    if not result_videos:
        print("[SadTalker] No video produced. Check SadTalker output for errors.")
    else:
        latest_video = max(result_videos, key=os.path.getctime)
        print("Bot (video): Previewing lipsynced video...")
        video_clip = VideoFileClip(latest_video)
        video_clip_resized = video_clip.resize(height=320)
        video_clip_resized.preview()
        video_clip_resized.close()
        video_clip.close()
    os.remove(AUDIO_PATH)

    end_time = time.time()
    print(f"Total reply+audio+video time: {round(end_time-start_time, 1)} seconds\n---")
