from tool_calling import run_agent

IN_SOURCE_EVAL_SET = [
    {"question": 'What are the two narratives that the May 22 Yield Radar says are pumping?',
        "must_contain": [["private inference"], ["base trenches"]]},
    {"question": 'In the May 22 Yield Radar, what approach are you taking to the strategies?',
        "must_contain": [["conservative"]]},
    {"question": 'What is the risk range of these strategies mentioned in the May 22 Yield Radar?',
        "must_contain": [["low to medium risk","low - medium","low to medium"]]},
    {"question": 'What is the expected yield of the “Flagship USDC SuperVault” strategy?',
        "must_contain": [["8 to 9%","~8-9%","8-9%","8 - 9%"]]},
    {"question": 'What is the notable auditor of the “Flagship USDC SuperVault”?',
        "must_contain": [["cantina"]]},
    {"question": 'How much yield are the sUP rewards contributing to the “Flagship USDC SuperVault”?',
        "must_contain": [["5.46%"]]},
    {"question": 'What are the two options to add liquidity to the pool in the “wnAUSD-wnUSDC-wnUSDT0” strategy?',
        "must_contain": [["proportional"], ["flexible"]]},
    {"question": 'What is the TVL of Neverland, the #2 native lending protocol on Monad?',
        "must_contain": [["$43M TVL","$43 million","$43 million TVL","$43M"]]},
    {"question": 'What percentage does the pool take as swap fees in the “wnAUSD-wnUSDC-wnUSDT0” strategy?',
        "must_contain": [["0.0005%"]]},
    {"question": 'In the Merkl incentives of the “wnAUSD-wnUSDC-wnUSDT0” strategy, which tokens are used to pay the rewards?',
        "must_contain": [["DUST"], ["WMON"]]},
    {"question": 'How much is Steakhouse Financial managing now in total assets?',
        "must_contain": [["$2B", "2B", "two billion", "2 billion", "$2 billion"]]},
    {"question": 'how do I loop USDT0 on Aave?',
        "must_contain": [["deBridge"], ["Monad"]]},
    {"question": 'what is the low risk stablecoin strategy on Avalanche?',
        "must_contain": [["USDT Vault"], ["Benqi"]]},
    {"question": '"What are the yield sources of the sUSDai"',
        "must_contain": [["US T-Bills", "T-Bills", "Treasury Bills"], ["Interest paid by GPU infrastructure borrowers", "GPU infrastructure borrowers"]]},
]

OUT_OF_SOURCE_EVAL_SET = [
    {"question": 'What is the max supply of BTC?',
        "must_not_contain": ["21000000", "21 million", "21M", "twenty-one million"]},
    {"question": 'When did Vitalik Buterin first publish the core Ethereum Whitepaper?',
        "must_not_contain": ["2013", "two thousands and thirteen"]},
    {"question": 'What is the trading time open of the crypto industry?',
        "must_not_contain": ["24/7", "24 hours per day 7 days per week", "around the clock"]},
    {"question": 'When was Bitcoin invented?',
        "must_not_contain": ["2008", "two thousands and eight"]},
    {"question": 'What will decide when the Bitcoin halving event happen?',
        "must_not_contain": ["210000 blocks", "210000", "210k", "210,000"]},
    {"question": 'When was the first official Bitcoin network transaction?',
        "must_not_contain": ["2009", "two thousands and nine"]},
    {"question": 'In the famous bitcoin pizza day, how many bitcoins did the guy use to purchase pizza n 2010?',
        "must_not_contain": ["10,000", "10000", "10k", "10K", "ten thousand"]},
    {"question": 'How many Satoshis (or sats) make up a single Bitcoin?',
        "must_not_contain": ["100000000", "100,000,000", "100 million", "100M", "one hundred million "]},
    {"question": 'What is the estimate year that the very last Bitcoin will be mined?',
        "must_not_contain": ["2140", "two thousands one hundred and forty"]},
    {"question": 'When did Ethereum officially launch?',
        "must_not_contain": ["2015", "two thousands and fifteen"]},
]

in_source_pass = 0
in_source_fail = 0
in_source_total = len(IN_SOURCE_EVAL_SET)

print(f"==========IN SOURCE EVAL RESULT==========\n")

for case in range (0,in_source_total):
    question = IN_SOURCE_EVAL_SET[case]["question"]
    must_contain = IN_SOURCE_EVAL_SET[case]["must_contain"]
    must_contain_clean = [[k.strip().lower() for k in sublist] for sublist in must_contain]
    response = run_agent(question).strip().lower()
    key_found = 0
    total_key = len(must_contain)
    missing_key = []
    for key in must_contain_clean:
        check = any(k in response for k in key)
        if check:
            key_found += 1
        else:
            missing_key.append(key)
    if key_found == total_key:
        result = "Pass ✅"
        missing_keys = None
        in_source_pass += 1
    elif key_found < total_key:
        result = "Fail ❌"
        in_source_fail += 1
    print(f"> 🍻🍻🍻 Q{case+1}:")
    print(f"# 🦖 Agent response: {response})")
    print(f"# 🦖 Final result: {key_found}/{total_key} -> {result} (missing: {missing_key})\n////////////\n")


print(f"--------------------------------")
print(f"> Overall rate: {in_source_pass}/{in_source_total} = {(in_source_pass/in_source_total):.2%}\n\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n\n")

out_of_source_pass = 0
out_of_source_fail = 0
out_of_source_total = len(OUT_OF_SOURCE_EVAL_SET)

print(f"==========OUT OF SOURCE EVAL RESULT==========\n")

for case in range (0,out_of_source_total):
    question = OUT_OF_SOURCE_EVAL_SET[case]["question"]
    must_not_contain = OUT_OF_SOURCE_EVAL_SET[case]["must_not_contain"]
    response = run_agent(question)
    total_key_check = len(must_not_contain)
    key_found = []
    for key in must_not_contain:
        if key.strip().lower() in response.strip().lower():
            key_found.append(key)
    if len(key_found) >= 1:
        result = "Fail ❌"
        out_of_source_fail += 1
    else:
        result = "Pass ✅"
        out_of_source_pass += 1
    print(f"> 🍻🍻🍻 Q{case+1}:")
    print(f"# 🦖 Agent response: {response})")
    print(f"# 🦖 Final result: {len(key_found)}/{total_key_check} -> {result} (key found: {key_found})\n////////////\n")


print(f"--------------------------------")
print(f"> Overall rate: {out_of_source_pass}/{out_of_source_total} = {(out_of_source_pass/out_of_source_total):.2%}")