from sentence_transformers import SentenceTransformer, util
import torch
import anthropic
import os
from dotenv import load_dotenv
from chunking import build_chunks, chunks_by_content

load_dotenv()

ANTHROPIC_KEY = os.getenv("ANTHROPIC_KEY")

client = anthropic.Anthropic(
    api_key=ANTHROPIC_KEY,
)

model = SentenceTransformer("all-MiniLM-L6-v2")

# files = build_chunks("yield_radar.txt")
files = chunks_by_content("yield_radar.txt")

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

for index, chunk in enumerate(files):
    print(f"=========CHUNK {index}=========\n")
    print(f"{chunk}\n")
    print(f"---------------------------------\n")