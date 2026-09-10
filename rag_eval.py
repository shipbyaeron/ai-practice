from sentence_transformers import SentenceTransformer, util
import os
from dotenv import load_dotenv
import anthropic
import torch
from chunking import build_chunks, chunks_by_content

load_dotenv()

ANTHROPIC_KEY = os.getenv("ANTHROPIC_KEY")

client = anthropic.Anthropic(
    api_key=ANTHROPIC_KEY,
)

model = SentenceTransformer("all-MiniLM-L6-v2")

files = chunks_by_content("yield_radar.txt")

files_embedded = model.encode(files)

# EVAL_SET = [
#     {"question": 'What are the yield sources of the “Flagship USDC SuperVault”?',
#         "expected_chunks": [1]},
#     {"question": 'How old is the “Flagship USDC SuperVault”?',
#         "expected_chunks": [1]},
#     {"question": 'What are the base yield of the “Flagship USDC SuperVault”?',
#         "expected_chunks": [1]},
#     {"question": 'What are the main exposures of the “Flagship USDC SuperVault”?',
#         "expected_chunks": [2]},
#     {"question": 'How many days does it take to unstake sUP rewards?',
#         "expected_chunks": [2]},
#     {"question": 'What are the three stable coins that are allowed to provide liquidity to the pool in the “wnAUSD-wnUSDC-wnUSDT0” strategy?',
#         "expected_chunks": [3]},
#     {"question": 'When will you know you have finished setting up the “wnAUSD-wnUSDC-wnUSDT0” strategy?',
#         "expected_chunks": [3]},
#     {"question": 'Who did the latest audit for Balancer?',
#         "expected_chunks": [3]},
#     {"question": 'What are the yield sources of the “wnAUSD-wnUSDC-wnUSDT0” strategy?',
#         "expected_chunks": [4, 5]},
#     {"question": 'How much yield does Merlk incentives contribute to the “wnAUSD-wnUSDC-wnUSDT0” strategy?',
#         "expected_chunks": [5]},
# ]

EVAL_SET = [
    {"question": 'What are the yield sources of the “Flagship USDC SuperVault”?',
        "expected_chunks": [5]},
    {"question": 'How old is the “Flagship USDC SuperVault”?',
        "expected_chunks": [4]},
    {"question": 'What are the base yield of the “Flagship USDC SuperVault”?',
        "expected_chunks": [5]},
    {"question": 'What are the main exposures of the “Flagship USDC SuperVault”?',
        "expected_chunks": [5]},
    {"question": 'How many days does it take to unstake sUP rewards?',
        "expected_chunks": [6]},
    {"question": 'What are the three stable coins that are allowed to provide liquidity to the pool in the “wnAUSD-wnUSDC-wnUSDT0” strategy?',
        "expected_chunks": [7]},
    {"question": 'When will you know you have finished setting up the “wnAUSD-wnUSDC-wnUSDT0” strategy?',
        "expected_chunks": [7]},
    {"question": 'Who did the latest audit for Balancer?',
        "expected_chunks": [8]},
    {"question": 'What are the yield sources of the “wnAUSD-wnUSDC-wnUSDT0” strategy?',
        "expected_chunks": [9]},
    {"question": 'How much yield does Merlk incentives contribute to the “wnAUSD-wnUSDC-wnUSDT0” strategy?',
        "expected_chunks": [9]},
]

expected_chunks_rank = []
total_pass = 0
total_fail = 0
for i in range (0, len(EVAL_SET)):
    question = EVAL_SET[i]["question"]
    question_embedded = model.encode(question)
    similarity = util.cos_sim(files_embedded, question_embedded)
    scores = similarity.squeeze()
    scores_list = [s.item() for s in scores]
    sorted_indices = torch.argsort(scores, descending=True).tolist()
    values, indices = torch.topk(scores, k=3)
    top_chunks_indices = [k.item() for k in indices]
    top_chunks_scores = [round(score.item(), 2) for score in values]
    expected_chunks = EVAL_SET[i]["expected_chunks"]
    missed_chunks = []
    total = len(expected_chunks)
    correct = 0
    for m in expected_chunks:
        if m in top_chunks_indices:
            correct += 1
        else:
            missed_chunks.append(m)
    print(f"==========QUESTION {i+1}==========")
    print(f"> Top chunks: {top_chunks_indices}")
    print(f"> Top chunks score: {top_chunks_scores}")
    print(f"> Correction rate: {correct}/{total}={(correct/total):.2%}")
    if correct == total:
        result = "Pass ✅"
        total_pass += 1
        print(f"> Retrieval result: {result}")
    elif correct < total:
        result = "Fail ❌"
        total_fail += 1
        print(f"> Retrieval result: {result}")
        # print(f"> Missed chunks: {missed_chunks}")
        # for mc in missed_chunks:
        #     print(f"> Expected chunk {mc} score = {scores_list[mc]:.2f}, rank {sorted_scores.index(scores_list[mc])+1}")
    else:
        print(f"We need more chunks to answer this question!")
    for e in expected_chunks:
        rank = sorted_indices.index(e) + 1
        expected_chunks_rank.append(rank)
        print(f"> Expected chunk {e} -> rank {rank} ({scores_list[e]:.2f})")
total_ques = total_pass + total_fail
mean_rank = 0
for r in expected_chunks_rank:
    mean_rank += r / len(expected_chunks_rank)
print(f"============OVERALL============")
print(f"> Pass rate: {total_pass}/{total_ques} = {(total_pass/total_ques):.2%}")
print(f"> Mean rank: {mean_rank}")