import os
import anthropic
import requests
import re

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
# USER_QUESTION = "Who created Bitcoin"
USER_QUESTION = "How do I loop USDT0 on Aave?"


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
    'Your knowledge is limited to the tools and the sources I give you.'
    'For ANY question, you MUST call the search_articles first and search for the information before you can conclude you do not know. Never refuse without searching first.'
    'If you do not do this, you are violating your role rules.'
    'For every questions, always use the search_articles tool and read all my research carefully from the top to the bottom to see if the information you need to answer the question is there.' 
    'You have to make sure you understand the meaning of the whole article and the question before answering any questions.'
    'If you do not find the information, read it the second time. Some information might be buried in my research since it is often a long article'
    'Only conclude you do not know once you scanned through all my research and there is no provided information about it.'
    
    '[CONTEXT]: '
    'You will receive a question about crypto in general.'

    '[INSTRUCTION]:'
    'Workflow: First you need to analyze the question, understand it deeply, then search for the information you need to answer the question.'
    'Overall: Use the tools you need, take the tool result to answer the question, and that is it.'
    'If the question relating to a coin price, you will need to take from the question the coin name, then use the tool to get the price. '
    'The price format: i) If the price > 1000, no digit. ii) From 1 to 1000, two digit precise. iii) From 0 to 1, always has 2 digits precise. '
    'If it has 0 after the ., it should be like this 0.023 or 0.00032.' 
    'If the question about portfolio, you will need to use the get_transaction tool. It will give you all the transactions log. '
    'If you need the pnl information of each position or the whole portfolio, use the get_pnl tool.' 

    '[CONSTRAINTS] Do not add any other type of information based on your training memory or general knowledge. ' 
    'If you do not have enough context to answer the question, I want the response only has 1 sentence which is "I do not have enough context to answer that question."'
    'If you could not find the information, your ENTIRE response should be that only one sentece. Nothing more before or after it.'
    'Otherwise, if you have enough context to answer it, do not hesitate to give more information (limit to the knowledge and tool result you have)'
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
        "description": ("Search for users' crypto research for any type of crypto questions regardless the topic."
        "It could be about companies, strategies, defi, protocols, KOLs, numbers, etc. Anything that is crypto-related."
        "Always try this first before saying you do not know."
        "The output of this tool is top few paragraphs that have the information you need."
        "Use only that information to answer the question"),
        "input_schema": {
            "type": "object",
            "properties": {
                "user_question": {
                    "type": "string",
                    "description": "the user's original question"
                },
            },
            "required": ["user_question"]
        },
    }
]

def run_agent(user_question, verbose=False, return_resources=False):
    messageList = [{"role": "user","content": user_question}]
    turn = 0
    sources_list = []
    while True:
        turn += 1
        if turn > 10:
            error_message = "There has been too many loops already :( Pls check the question logic or tools and run again."
            if return_resources:
                return (error_message, [])
            else:
                return f"{error_message}"

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            tools=tools,
            system=SYSTEM_PROMPT,
            messages=messageList,
        )

        if verbose:
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
                            pattern = r"> Source:\s*(.*?)\s*> Chunk content:"
                            matches = re.findall(pattern, api_result, re.DOTALL) 
                            for match in matches:
                                if match not in sources_list:
                                    sources_list.append(match)
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
            if return_resources:
                return (final_text, sources_list)
            else:
                return final_text
        else:
            error = response.stop_reason
            error_message = f"The program has stopped. The stop reason is {error}."
            if return_resources:
                return (error, [])
            else:
                return f"{error_message}"

def main():
    final_text = run_agent(USER_QUESTION, verbose=True, return_resources=False)
    print(final_text)

if __name__ == "__main__":
    main()