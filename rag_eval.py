from sentence_transformers import SentenceTransformer, util
import os
from dotenv import load_dotenv
import anthropic
import torch
from chunking import build_chunks, chunks_by_section
from rag_basic import chunks_list, chunks_embedded, expand_query

load_dotenv()

ANTHROPIC_KEY = os.getenv("ANTHROPIC_KEY")

client = anthropic.Anthropic(
    api_key=ANTHROPIC_KEY,
)

model = SentenceTransformer("all-MiniLM-L6-v2")

EVAL_SET_CHUNKS_BY_SECTION = [
    {"question": 'What are the yield sources of the “Flagship USDC SuperVault”?',
        "must_contain": [["base yield"], ["sUP incentives"]]},
    {"question": 'How old is the “Flagship USDC SuperVault”?',
        "must_contain": [["5 months", "five months"]]},
    {"question": 'What are the base yield of the “Flagship USDC SuperVault”?',
        "must_contain": [["3.29%", "three point twenty nine percent"]]},
    {"question": 'What are the main exposures of the “Flagship USDC SuperVault”?',
        "must_contain": [["Gauntlet USDC Prime", "Gauntlet USDC"], ["Steakhouse USDC on Morpho", "Steakhouse USDC"]]},
    {"question": 'How many days does it take to unstake sUP rewards?',
        "must_contain": [["14-day", "14 days", "fourteen days"]]},
    {"question": 'What are the three stable coins that are allowed to provide liquidity to the pool in the “wnAUSD-wnUSDC-wnUSDT0” strategy?',
        "must_contain": [["USDT0"], ["AUSD"], ["USDC"]]},
    {"question": 'When will you know you have finished setting up the “wnAUSD-wnUSDC-wnUSDT0” strategy?',
        "must_contain": [["BPT", "Balancer Pool Tokens"]]},
    {"question": 'Who did the latest audit for Balancer?',
        "must_contain": [["Centora"]]},
    {"question": 'What are the yield sources of the “wnAUSD-wnUSDC-wnUSDT0” strategy?',
        "must_contain": [["Swap fees"], ["Yield-bearing stablecoins", "yield bearing stablecoins"], ["Merkl incentives", "Merkl"]]},
    {"question": 'How much yield does Merlk incentives contribute to the “wnAUSD-wnUSDC-wnUSDT0” strategy?',
        "must_contain": [["10-11%", "ten to eleven percent", "10 to 11%", "10 - 11%", "10% to 11%"]]},
    {"question": '"What are the yield sources of the sUSDai"',
        "must_contain": [["US T-Bills", "T-Bills", "Treasury Bills"], ["Interest paid by GPU infrastructure borrowers", "GPU infrastructure borrowers"]]},
]

chunks_clean = [chunk.strip().lower() for chunk in chunks_list]

TOP_K = 3
deepest_chunks_rank = []
total_pass = 0
total_fail = 0

missing_facts_count = 0
out_of_top_k_count = 0

for i in range (0, len(EVAL_SET_CHUNKS_BY_SECTION)):
    question = EVAL_SET_CHUNKS_BY_SECTION[i]["question"]
    must_contain = EVAL_SET_CHUNKS_BY_SECTION[i]["must_contain"]
    must_contain_clean = [[k.strip().lower() for k in sublist] for sublist in must_contain]
    question_embedded = model.encode(expand_query(question))
    similarity = util.cos_sim(chunks_embedded, question_embedded)
    scores = similarity.squeeze()

    # List of the similarity scores of chunks (in the order of the original article)
    scores_list = [s.item() for s in scores]
    # The sorted indices by similarity scores (indices is also in the order of the original article)
    sorted_indices = torch.argsort(scores, descending=True).tolist()

    indices = torch.topk(scores, k=TOP_K).indices
    top_k_indices = [k.item() for k in indices]

    values = torch.topk(scores, k=TOP_K).values
    top_k_scores = [f"{k.item():.2f}" for k in values]

    key_found_infor = []
    missing_keys = []

    for key in must_contain_clean:
        for rank, chunk_index in enumerate(sorted_indices):
            check = any(k in chunks_clean[chunk_index] for k in key)
            if check:
                key_found_infor.append({
                        "key_detail": key,
                        "chunk_index": chunk_index,
                        "chunk_rank": rank + 1
                    })
                break

    total_keys_required = len(must_contain_clean)
    total_key_found = len(key_found_infor)

    key_found_detail_list = [key_found_infor[i]["key_detail"] for i in range (0, total_key_found)]
    key_found_rank_list = [key_found_infor[i]["chunk_rank"] for i in range (0, total_key_found)]

    for key in must_contain_clean:
        if key not in key_found_detail_list:
            missing_keys.append(key)

    if key_found_rank_list:
        deepest_rank = max(key_found_rank_list)
        deepest_chunks_rank.append(deepest_rank)
    else:
        raise ValueError("Zero keys found. Pls check the question or the must_contain keys list again!")

    if deepest_rank <= TOP_K and total_key_found == total_keys_required:
        result = "Pass ✅"
        total_pass += 1
    else:
        result = "Fail ❌"
        total_fail += 1

    print(f"==========QUESTION {i+1}==========")
    print(f">> Keys requires ({total_keys_required}):", end="")
    for key in must_contain:
        print(f" {key} |", end="")
    print()
    for num in range(0, len(must_contain_clean)):
        fact = must_contain_clean[num]
        if fact in key_found_detail_list:
            for m in range(0, len(key_found_infor)):
                if key_found_infor[m]["key_detail"] == fact:
                    chunk_rank = key_found_infor[m]["chunk_rank"]
                    break
            print(f"# Key {num+1}: rank {chunk_rank}")
        else:
            print(f"# Key {num+1}: rank N/A (key not found)")
    print(f">> Deepest rank: {deepest_rank}")
    print(f">> Result: {result}")
    if result == "Fail ❌":
        if total_key_found == total_keys_required:
            out_of_top_k_count += 1
            print(f"# Reason on fail: all found but {deepest_rank} > k({TOP_K})")
        else:
            missing_facts_count += 1
            print(f"# Reason on fail: missing {missing_keys}")

total_ques = total_pass + total_fail
mean_rank = 0
for r in deepest_chunks_rank:
    mean_rank += r / len(deepest_chunks_rank)
print(f"============OVERALL============")
print(f"> Pass rate: {total_pass}/{total_ques} = {(total_pass/total_ques):.2%}")
print(f"> Mean rank: {mean_rank}")
print(f"> Fails by type: {missing_facts_count} missing-fact, {out_of_top_k_count} out-of-top-k")