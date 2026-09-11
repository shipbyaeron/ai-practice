import os
import anthropic
import requests

from dotenv import load_dotenv
from rag_basic import search_articles

load_dotenv()

class ToolError(Exception):
    pass

def get_price(coin_id):
    url = f"{API_URL}/price/{coin_id}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
    else:
        raise ToolError(f"Something went wrong with the get_price API endpoint :( Status code: {response.status_code}.")
    return data

def get_transactions():
    url = f"{API_URL}/transactions"
    headers = {"x-api-key": APP_API_KEY}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
    else:
        raise ToolError(f"Something went wrong with the get_transactions API endpoint :( Status code: {response.status_code}.")
    return data   

def get_pnl():
    url = f"{API_URL}/pnl"
    headers = {"x-api-key": APP_API_KEY}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
    else:
        raise ToolError(f"Something went wrong with the get_pnl API endpoint :( Status code: {response.status_code}.")
    return data

# USER_QUESTION = "Am I up or down on my portfolio?"
# USER_QUESTION = "What does the Yield Radar say about the Balancer audit?"
# USER_QUESTION = "How many strategies are there in the May 22 Yield Radar? What are they and which chains are the strategies running on?"
USER_QUESTION = "Who created Bitcoin"


ANTHROPIC_KEY = os.getenv("ANTHROPIC_KEY")
API_URL = os.getenv("API_URL")
APP_API_KEY = os.getenv("APP_API_KEY")
TOOL_FUNCTIONS = {
    "get_price": get_price,
    "get_transactions": get_transactions,
    "get_pnl": get_pnl,
    "search_articles": search_articles
}

SYSTEM_PROMPT = (
    '[ROLE]: You are my assisstant helping me answer my questions.' 
    'Your knowledge is limited to the tools and the sources I give you. Nothing else.'
    'It should be fine to say you do not know something if the tools and sources I give you do not cover the information you need to answer it.'
    '[CONTEXT]: You will receive a question about crypto in general.'
    '[INSTRUCTION]: If the question relating to a coin price, you will need to take from the question the coin name, then use the tool to get the price. '
    'The price format: i) If the price > 1000, no digit. ii) From 1 to 1000, two digit precise. iii) From 0 to 1, always has 2 digits precise. '
    'If it has 0 after the ., it should be like this 0.023 or 0.00032.' 
    'If the question about portfolio, you will need to use the get_transaction tool. It will give you all the transactions log. '
    'If you need the pnl information of each position or the whole portfolio, use the get_pnl tool.'
)

client = anthropic.Anthropic(
    api_key=ANTHROPIC_KEY,
)    

tools = [
    {
        "name": "get_price",
        "description": "Get the price of a coin in USD",
        "input_schema": {
            "type": "object",
            "properties": {
                "coin_id": {
                    "type": "string",
                    "description": "CoinGecko coin id. For example: bitcoin, ethereum, solana"
                }
            },
            "required": ["coin_id"]
        },
    },
    {
        "name": "get_transactions",
        "description": ("Whenever you receive a question about the portfolio or all transansactions log,  you will need to use this tool."
        "The output should be all the transactions log of all coins."
        "For each transaction, there are 5 information needed: coin (the coin_id with Coingecko format), action, amount, price, and total."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_pnl",
        "description": ("Whenever you need to know about the PnL of each position and the overall portfolio, use this tool."
        "This tool will then give you three different information at the same time."
        "i) Detailed information of each position: total_holding (total current holding of that coin), avg_entry_price, current_market_price of the coin, percentage (how much it's up or down in %), and pnl in USD"
        "ii) Overall information of the portfolio: holding_cost (the cost basis of current holding), portfolio_current_value (the market value of the porfolio), percentage (how much the whole porfolio is up or down in %), and pnl in USD"
        "iii) The list of invalid coins: These are coins that are not be able to request the price from Coingecko. This could be because of the wrong typo, doesn't exist, etc"
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "search_articles",
        "description": ("Only use this tool when you need to search for information in my own Digest crypto writing."
        "The output of this tool is top few paragraphs that have the information you need."
        "Use only that information to answer the question"),
        "input_schema": {
            "type": "object",
            "properties": {
                "user_question": {
                    "type": "string",
                    "description": "the user's original question"
                }
            },
            "required": ["user_question"]
        },
    }
]

messageList = [{"role": "user","content": USER_QUESTION}]

turn = 0

while True:
    turn += 1
    if turn > 10:
        print(f"There has been too many loops already :( Pls check the question logic or tools and run again.")
        break

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        tools=tools,
        system=SYSTEM_PROMPT,
        messages=messageList,
    )

    print(f"turn {turn} stop_reason={response.stop_reason}")
    
    if response.stop_reason == "tool_use": 
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                if block.name not in TOOL_FUNCTIONS:
                    api_result = f"Unknown tool: {block.name}"
                    is_error = True
                else:
                    fn = TOOL_FUNCTIONS[block.name] 
                    try:
                        api_result = fn(**block.input)
                        is_error = False
                    except ToolError as e:
                        api_result = str(e)
                        is_error = True
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(api_result),
                    "is_error": is_error
                })
        messageList.append({"role": "assistant", "content": response.content})
        messageList.append({"role": "user", "content": tool_results})

    elif response.stop_reason == "end_turn":
        texts = []
        for block in response.content:
            if block.type == "text":
                texts.append(block.text)
        final_text = "".join(texts)
        print(final_text)
        break

    else:
        error = response.stop_reason
        print (f"The program has stopped. The stop reason is {error}.") 
        break