from dotenv import load_dotenv
import os
from langchain.agents import create_agent
from langchain.tools import tool 
from langchain_core.messages import HumanMessage 
from langchain_mistralai import ChatMistralAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch
load_dotenv()



tavily=TavilyClient()

@tool
def search(query:str)-> str:
    """
    Tool that searches on the Internet
    Args
        query: The query to search for 
    Returns: 
        The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)

llm=ChatMistralAI()
tools=[TavilySearch()]
agent=create_agent(model=llm,tools=tools)

def main():
    print("Hello from langchain-course!")
    result=agent.invoke({"messages":HumanMessage(content=("search for 3 jobs posting for ai engineer that handle langhcain in the bay area on linkedin and list their details"))})
    print(result)

if __name__ == "__main__":
    main()
