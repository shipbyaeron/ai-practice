from sentence_transformers import SentenceTransformer, util
import torch
import anthropic
import os
from dotenv import load_dotenv
from chunking import chunks_by_content

load_dotenv()

ANTHROPIC_KEY = os.getenv("ANTHROPIC_KEY")
FILE_PATH = "yield_radar.txt"
NUM_TOP_CHUNKS = 3
user_question = "What is the price of Bitcoin today?"

client = anthropic.Anthropic(
    api_key=ANTHROPIC_KEY,
)

model = SentenceTransformer("all-MiniLM-L6-v2")
chunks = chunks_by_content(FILE_PATH)
chunks_embedded = model.encode(chunks)

def search_articles(user_question: str):
    question_embedded = model.encode(user_question)
    all_similarity = util.cos_sim(chunks_embedded, question_embedded)
    scores_data = all_similarity.squeeze()
    indices = torch.topk(scores_data, k=NUM_TOP_CHUNKS).indices
    top_chunks = []
    for i in indices:
        top_chunks.append(chunks[i.item()])
    context = "\n-----------\n".join(top_chunks)
    return context

def main():
    context = search_articles(user_question)
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
            "content": user_question}
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
    for index, chunk in enumerate(chunks):
        print(f"=========CHUNK {index}=========\n")
        print(f"{chunk}\n")
        print(f"---------------------------------\n")

if __name__ == "__main__":
    main()