from sentence_transformers import SentenceTransformer, util
import torch
import anthropic
import os
from dotenv import load_dotenv
from chunking import chunks_by_section

load_dotenv()

ANTHROPIC_KEY = os.getenv("ANTHROPIC_KEY")
NUM_TOP_CHUNKS = 3
user_question = "What is the price of Bitcoin today?"

FOLDER_NAME = "articles"
FILE_NAME = ["yield_radar_may08.txt", "yield_radar_may22.txt", "yield_radar_june19.txt", "yield_radar_july09.txt"]
FILE_PATH = []
for file_name in FILE_NAME:
    file_path = os.path.join(FOLDER_NAME, file_name)
    FILE_PATH.append(file_path)

client = anthropic.Anthropic(
    api_key=ANTHROPIC_KEY,
)

model = SentenceTransformer("all-MiniLM-L6-v2")

chunks_infor = []

for filepath in FILE_PATH:
    article_title, filepath_chunks = chunks_by_section(filepath)
    for chunk in filepath_chunks:
        chunks_infor.append({
            "chunk_content": chunk,
            "file_path": filepath,
            "article_title": article_title
        })

# List of chunks of every articles without the file_path
chunks_list = [chunks_infor[i]["chunk_content"] for i in range (0, len(chunks_infor))]

chunks_embedded = model.encode(chunks_list)

def expand_query(user_question: str) -> str:
    SYSTEM_PROMPT = (
        "[ROLE]:"
        "You will receive a question about crypto."
        "Your main role is to rewrite the question so that in the next step, the model's retrieval could rank the chunks better because it could understand the question better."

        "[INSTRUCTION]:"
        "You will imagine what the answer looks like for that question. "
        "The answer you think about will have the terms that haven't existed in the original question. Include those terms in the rewrited one."
        "You have to keep the original idea and intent of the question."
        "You can use the domain synonyms and keywords that would actually appeared in my research."
        "Your response should only be the rewrited question. Nothing else."
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
        final_question = " ".join(text)
        return final_question
    else:
        error = response.stop_reason
        raise ValueError(f"Something went wrong with the expand_query function. Stop reason error: {error}. Pls check!")

def search_articles(user_question: str):
    question_embedded = model.encode(expand_query(user_question))
    all_similarity = util.cos_sim(chunks_embedded, question_embedded)
    scores_data = all_similarity.squeeze()
    indices = torch.topk(scores_data, k=NUM_TOP_CHUNKS).indices
    top_chunks = []
    sources_list = []
    for indice in indices:
        idx = indice.item()
        chunk = chunks_infor[idx]["chunk_content"]  
        source = chunks_infor[idx]["article_title"]
        sources_list.append(source)
        top_chunks.append(f"> Source: {source}\n> Chunk content: {chunk}")
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
        print(f"Response: {final_text}")
    else:
        error = response.stop_reason
        print(f"The program has stopped. The stop reason: {error}.")
    for index, chunk in enumerate(chunks_list):
        print(f"=========CHUNK {index}=========\n")
        print(f"{chunk}\n")
        print(f"---------------------------------\n")

if __name__ == "__main__":
    print(expand_query("When will you know you have finished setting up the “wnAUSD-wnUSDC-wnUSDT0” strategy?"))
    # main()