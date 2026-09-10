from sentence_transformers import SentenceTransformer, util
import torch
import anthropic
import os
from dotenv import load_dotenv
from chunking import build_chunks

load_dotenv()

ANTHROPIC_KEY = os.getenv("ANTHROPIC_KEY")

client = anthropic.Anthropic(
    api_key=ANTHROPIC_KEY,
)

model = SentenceTransformer("all-MiniLM-L6-v2")

files = build_chunks("yield_radar.txt")

# EVAL_SET = [
#     {"question": 'What are the yield sources of the “Flagship USDC SuperVault”?',
#         "expected_chunks": [1],
#         "must_contain": ["base yield", "sUP incentives"]},
#     {"question": 'How old is the “Flagship USDC SuperVault”?',
#         "expected_chunks": [1],
#         "must_contain": ["5 months"]},
#     {"question": 'What are the base yield of the “Flagship USDC SuperVault”?',
#         "expected_chunks": [1],
#         "must_contain": ["3.29"]},
#     {"question": 'What are the main exposures of the “Flagship USDC SuperVault”?',
#         "expected_chunks": [2],
#         "must_contain": ["Gauntlet USDC Prime", "Steakhouse", "Morpho"]},
#     {"question": 'How many days does it take to unstake sUP rewards?',
#         "expected_chunks": [2],
#         "must_contain": ["14-day"]},
#     {"question": 'What are the three stable coins that are allowed to provide liquidity to the pool in the “wnAUSD-wnUSDC-wnUSDT0” strategy?',
#         "expected_chunks": [3],
#         "must_contain": ["USDT0", "AUSD", "USDC"]},
#     {"question": 'When will you know you have finished setting up the “wnAUSD-wnUSDC-wnUSDT0” strategy?',
#         "expected_chunks": [3],
#         "must_contain": ["BPT"]},
#     {"question": 'Who did the latest audit for Balancer?',
#         "expected_chunks": [3],
#         "must_contain": ["Centora"]},
#     {"question": 'What are the yield sources of the “wnAUSD-wnUSDC-wnUSDT0” strategy?',
#         "expected_chunks": [4, 5],
#         "must_contain": ["swap fees", "yield-bearing stablecoins", "merkl incentives"]},
#     {"question": 'How much yield does Merlk incentives contribute to the “wnAUSD-wnUSDC-wnUSDT0” strategy?',
#         "expected_chunks": [5],
#         "must_contain": ["10","11"]},
# ]

EVAL_SET = [
    {"question": 'What are the yield sources of the “Flagship USDC SuperVault”?',
        "expected_chunks": [5],
        "must_contain": ["base yield", "sUP incentives"]},
    {"question": 'How old is the “Flagship USDC SuperVault”?',
        "expected_chunks": [4],
        "must_contain": ["5 months"]},
    {"question": 'What are the base yield of the “Flagship USDC SuperVault”?',
        "expected_chunks": [5],
        "must_contain": ["3.29"]},
    {"question": 'What are the main exposures of the “Flagship USDC SuperVault”?',
        "expected_chunks": [5],
        "must_contain": ["Gauntlet USDC Prime", "Steakhouse", "Morpho"]},
    {"question": 'How many days does it take to unstake sUP rewards?',
        "expected_chunks": [6],
        "must_contain": ["14-day"]},
    {"question": 'What are the three stable coins that are allowed to provide liquidity to the pool in the “wnAUSD-wnUSDC-wnUSDT0” strategy?',
        "expected_chunks": [7],
        "must_contain": ["USDT0", "AUSD", "USDC"]},
    {"question": 'When will you know you have finished setting up the “wnAUSD-wnUSDC-wnUSDT0” strategy?',
        "expected_chunks": [7],
        "must_contain": ["BPT"]},
    {"question": 'Who did the latest audit for Balancer?',
        "expected_chunks": [8],
        "must_contain": ["Centora"]},
    {"question": 'What are the yield sources of the “wnAUSD-wnUSDC-wnUSDT0” strategy?',
        "expected_chunks": [9],
        "must_contain": ["swap fees", "yield-bearing stablecoins", "merkl incentives"]},
    {"question": 'How much yield does Merlk incentives contribute to the “wnAUSD-wnUSDC-wnUSDT0” strategy?',
        "expected_chunks": [9],
        "must_contain": ["10","11"]},
]

files_embedded = model.encode(files)

total_pass = 0
total_fail = 0

print(f"==========FINAL RESULT==========")

for case in range (0,len(EVAL_SET)):
    question = EVAL_SET[case]["question"]
    expected_chunks = EVAL_SET[case]["expected_chunks"]
    must_contain = EVAL_SET[case]["must_contain"]
    context_list = [files[m] for m in expected_chunks]

    # context = "\n-----------\n".join(top_3_chunks)
    context = "\n-----------\n".join(context_list)

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
        final_text = " ".join(text).strip().lower()
    else:
        error = response.stop_reason
        print(f"The program has stopped. The stop reason: {error}.")

    key_found = 0
    total_key = len(must_contain)
    missing_key = []

    for key in must_contain:
        if key.strip().lower() in final_text:
            key_found += 1
        else:
            missing_key.append(key)

    if key_found == total_key:
        result = "Pass ✅"
        missing_key = None
        total_pass += 1
    elif key_found < total_key:
        result = "Fail ❌"
        total_fail += 1
    else:
        result = "There are too many chunks used than we need!"
    
    print(f"> Q{case+1}: {key_found}/{total_key} -> {result} (missing: {missing_key})")

total = total_pass + total_fail

print(f"--------------------------------")
print(f"> Overall rate: {total_pass}/{total} = {(total_pass/total):.2%}")