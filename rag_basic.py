from sentence_transformers import SentenceTransformer, util
import torch
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_KEY = os.getenv("ANTHROPIC_KEY")

client = anthropic.Anthropic(
    api_key=ANTHROPIC_KEY,
)

model = SentenceTransformer("all-MiniLM-L6-v2")

def word_count(m):
    words = m.strip().split(" ")
    return len(words)

with open("yield_radar.txt", "r", encoding="utf-8-sig") as f:
    content = f.read()

files = []

sections = content.split("—--------------------")

for s in sections:
    small_chunks = [line for line in s.split("\n") if line.strip()]
    final_chunks = []
    for k in small_chunks:
        if word_count(k) < 150:
            final_chunks.append(k)
        else:
            sentences = k.split(".")
            final_chunks.extend(sentences)
    title = small_chunks[0]
    len_small = len(final_chunks)
    i = 1
    while i < len_small:
        current_group = [final_chunks[i]]
        current_length = word_count(final_chunks[i])
        j = i + 1
        while j < len_small:
            next_length = word_count(final_chunks[j])
            if current_length + next_length > 150:
                break
            current_group.append(final_chunks[j])
            current_length += next_length
            j += 1
        merged_text = " ".join(current_group)
        text_to_embedded = f"{title}\n\n{merged_text}"
        files.append(text_to_embedded)
        i = j

# question = "How many audits has Superform done?"
question = "What is the price of Bitcoin today?"

files_embedded = model.encode(files)

question_embedded = model.encode(question)

all_similarity = util.cos_sim(files_embedded, question_embedded)

scores = all_similarity.squeeze()

values, indices = torch.topk(scores, k=3)

top_3_chunks = []

for i in indices:
    top_3_chunks.append(files[i.item()])

context = "\n-----------\n".join(top_3_chunks)

SYSTEM_PROMPT = (
    f"[ROLE]: You're a blockchain expert. "
    f"[CONTEXT]: {context}"
    f"[INSTRUCTION]: You will only use the context you have to answer the question. If the information is not there, say don't know. " 
    f"Never use any of the outside documents or knowledge" 
)

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1024,
    system=SYSTEM_PROMPT,
    messages=[
        {"role": "user",
        "content": question}
    ]
)

if response.stop_reason == "end_turn":
    text = []
    for block in response.content:
        if block.type == "text":
            text.append(block.text)
    final_text = " ".join(text)
    print(final_text)
else:
    error = response.stop_reason
    print(f"The program has stopped. The stop reason: {error}.")