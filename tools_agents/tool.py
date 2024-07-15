import os 
from langchain_community.tools.tavily_search import TavilySearchResults
def get_profile_url(name:str):
    """search for linkedin and twitter profile pages"""
    search=TavilySearchResults()
    res=search.run(f"{name}")
    return res[0]["url"]
