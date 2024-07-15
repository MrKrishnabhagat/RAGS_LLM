import os

from tool import get_profile_url
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import (create_react_agent,AgentExecutor)
from langchain import hub
from dotenv import load_dotenv
load_dotenv()
def lookup(name:str):
    llm=ChatOpenAI(temperature=0,
                   model_name="gpt-3.5-turbo"
                   )
    template="""given the full name {name_of_person} i want u to get me a link to their linkedin profile page
    your answer should contain only a URL"""
    prompt_template=PromptTemplate(template=template,input_variables=["name_of_person"])
    tools_for_agent=[Tool(name="search google for linkedin profile page",
                          func=get_profile_url,
                          description="useful for when you need to get linkedin Page URL"
                          )]
    react_prompt=hub.pull("hwchase17/react")
    agent=create_react_agent(llm=llm,tools=tools_for_agent,prompt=react_prompt)
    agent_executor=AgentExecutor(agent=agent,tools=tools_for_agent,verbose=True)
    result=agent_executor.invoke(input={"input":prompt_template.format_prompt(name_of_person=name)})
    linkedin_profile_url=result["output"]
    return linkedin_profile_url






if __name__=="__main__":
    linkedin_url=lookup(name="dhruv bhagat taptalent ")
    print(linkedin_url)
