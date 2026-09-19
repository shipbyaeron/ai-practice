from sentence_transformers import SentenceTransformer, util
import torch
import anthropic
import os
from dotenv import load_dotenv
from chunking import chunks_by_section
import boto3
import math

load_dotenv()

ANTHROPIC_KEY = os.getenv("ANTHROPIC_KEY")
NUM_TOP_CHUNKS = 3
user_question = "What is the price of Bitcoin today?"

S3_BUCKET_NAME = os.getenv("S3_BUCKET")
FOLDER_NAME = "articles"

s3 = boto3.client("s3")

FILE_PATH = []
FILE_NAME = ["yield_radar_may08.txt", "yield_radar_may22.txt", "yield_radar_june19.txt", "yield_radar_july09.txt"]

os.makedirs(FOLDER_NAME, exist_ok=True)

for file_name in FILE_NAME:
    file_path = os.path.join(FOLDER_NAME, file_name)
    s3.download_file(S3_BUCKET_NAME, file_name, file_path)
    
    FILE_PATH.append(file_path)

client = anthropic.Anthropic(
    api_key=ANTHROPIC_KEY,
)

model = SentenceTransformer("all-MiniLM-L6-v2")

chunks_infor = []
report_list = []

for file_path in FILE_PATH:
    article_title, filepath_chunks = chunks_by_section(file_path)
    file_name = os.path.basename(file_path)
    name_only, extension = os.path.splitext(file_name)
    # Split the file name by _ and take the last part (the date)
    date = name_only.split("_")[-1]
    report_list.append(date)
    for chunk in filepath_chunks:
        chunks_infor.append({
            "chunk_content": chunk,
            "file_path": file_path,
            "article_title": article_title,
            "report": date
        })

# List of chunks of every articles without the file_path
chunks_list = [chunks_infor[i]["chunk_content"] for i in range (0, len(chunks_infor))]

chunks_embedded = model.encode(chunks_list)

TOTAL_NUM_CHUNKS = len(chunks_infor)

punctuation_list = [
    '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
    ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~'
]

document_freqs = {}

for idx, chunk in enumerate(chunks_list):
    term_freqs = {}
    terms = chunk.split()
    chunk_length = len(terms)
    terms_nodup = []
    for term in terms:
        clean_term = term.lower()
        clean_term = clean_term.strip("".join(punctuation_list))
        if clean_term not in term_freqs:
            term_freqs[clean_term] = 1
        else:
            term_freqs[clean_term] += 1
        if clean_term not in terms_nodup:
            terms_nodup.append(clean_term)
    for te in terms_nodup:
        if te not in document_freqs:
            document_freqs[te] = 1
        else:
            document_freqs[te] += 1
    chunks_infor[idx]["term_freqs"] = term_freqs
    chunks_infor[idx]["chunk_length"] = chunk_length

AVGDL = 0

for m in range(0, TOTAL_NUM_CHUNKS):
    AVGDL += chunks_infor[m]["chunk_length"] / TOTAL_NUM_CHUNKS

K1 = 1.5
B = 0.75

def keyword_search(user_question: str):
    question_terms = user_question.split()
    question_clean_term = []
    overall_scores = []
    term_idf = {}
    for term in question_terms:
        clean_term = term.lower()
        clean_term = clean_term.strip("".join(punctuation_list))
        if clean_term not in question_clean_term:
            question_clean_term.append(clean_term)
    for term in question_clean_term:
        if term in document_freqs:
            n_t = document_freqs[term]  # n(t) for term t  
            # only calculate and add to the dict the term that existed in the corpus
            # if it exist in the question but not in the corpus, just leave it. we won't calculate its score either
            term_idf[term] = math.log((TOTAL_NUM_CHUNKS - n_t + 0.5) / (n_t + 0.5) + 1)
    for seri in range(0, TOTAL_NUM_CHUNKS):
        chunk_detail = chunks_infor[seri]  # chunk D detail
        score_ques_D = 0
        for term in question_clean_term:
            if term in chunk_detail["term_freqs"]:
                f_t_D = chunk_detail["term_freqs"][term] # f(t,D) for term t in chunk D
                chunk_D_length = chunk_detail["chunk_length"]   # |D| 
                IDF_t = term_idf[term]
                score_t_D = (IDF_t * (f_t_D * (K1 + 1)) / (f_t_D + K1 * (1 - B + B * chunk_D_length / AVGDL))) 
                score_ques_D += score_t_D
        overall_scores.append({
            "chunk_infor_indice": seri,
            "chunk_content": chunk_detail["chunk_content"],
            "article_title": chunk_detail["article_title"],
            "score": score_ques_D
        })
    sorted_chunks = sorted(overall_scores, key=lambda x: x["score"], reverse=True)
    keyword_sorted = {}
    for s in range (0, len(sorted_chunks)):
        idx = sorted_chunks[s]["chunk_infor_indice"]
        if idx not in keyword_sorted:
            keyword_sorted[idx] = s + 1
    # top_k_chunks = []
    # for s in range(0,NUM_TOP_CHUNKS):
    #     source = sorted_chunks[s]["article_title"]
    #     chunk_content = sorted_chunks[s]["chunk_content"]
    #     top_k_chunks.append(f"> Source: {source}\n> Chunk content: {chunk_content}")
    # context = "\n-----------\n".join(top_k_chunks)
    return keyword_sorted

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

def semantic_search(user_question: str):
    question_embedded = model.encode(expand_query(user_question))
    all_similarity = util.cos_sim(chunks_embedded, question_embedded)
    scores_data = all_similarity.squeeze()
    indices_all = torch.sort(scores_data, descending=True).indices
    semantic_sorted = {}
    for num in range(0, len(indices_all)):
        idx = indices_all[num].item()
        if idx not in semantic_sorted:
            semantic_sorted[idx] = num + 1
    return semantic_sorted

RRF_K = 60

def hybrid_search(user_question: str, report=None):
    rrf_list = []
    keyword_sorted_list = keyword_search(user_question)
    semantic_sorted_list = semantic_search(user_question)
    for idx in range(0, TOTAL_NUM_CHUNKS):
        keyword_rank = keyword_sorted_list[idx]
        semantic_rank = semantic_sorted_list[idx]
        rrf_score = 1/(RRF_K + keyword_rank) + 1/(RRF_K + semantic_rank)
        rrf_list.append({
            "chunk_infor_indice": idx,
            "chunk_content": chunks_infor[idx]["chunk_content"],
            "chunk_source": chunks_infor[idx]["article_title"],
            "report": chunks_infor[idx]["report"],
            "keyword_rank": keyword_rank,
            "semantic_rank": semantic_rank,
            "rrf_score": rrf_score
        })
    sorted_rrf_list = sorted(rrf_list, key=lambda x: x["rrf_score"], reverse=True)
    if report:
        report = report.strip().lower()
        if report not in report_list:
            return sorted_rrf_list
        else:
            sorted_filtered_rrf_list = []
            for r in range(0, len(sorted_rrf_list)):
                if sorted_rrf_list[r]["report"] == report:
                    sorted_filtered_rrf_list.append(sorted_rrf_list[r])
            return sorted_filtered_rrf_list
    else:
        return sorted_rrf_list

def search_articles(user_question: str, report=None):
    sorted_rrf_list = hybrid_search(user_question, report)[:NUM_TOP_CHUNKS]
    top_k_chunks = []
    for s in range(0, len(sorted_rrf_list)):
        source = sorted_rrf_list[s]["chunk_source"]
        chunk_content = sorted_rrf_list[s]["chunk_content"]
        top_k_chunks.append(f"> Source: {source}\n> Chunk content: {chunk_content}")
    context = "\n-----------\n".join(top_k_chunks)
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
    result = hybrid_search('Who did the latest audit for Balancer?', report=None)
    print(result)
    # main()