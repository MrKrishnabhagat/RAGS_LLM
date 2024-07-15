
import os
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from third_parties.linkedin import scrape_linkedin_profile
from linkedin_lookup_agent import lookup as linkedin_lookup_agent
from output_parsers import summary_parser

def icebreak(name:str):
    linkedin_username=linkedin_lookup_agent(name=name)
    linkedin_data=scrape_linkedin_profile(linkedin_url=linkedin_username)   
    summary_template="""given the information{information} about a person i want u to create :
    1.a short summary
    2.interesting facts about them
    \n{format_instructions}
    """
    summary_prompt_template=PromptTemplate(input_variables=["information"],template=summary_template,
                                           partial_variables={"format_instructions":summary_parser.get_format_instructions()})
    llm=ChatOpenAI(temperature=0,model_name="gpt-3.5-turbo")
    chain=summary_prompt_template|llm
    # chain=summary_prompt_template|llm|summary_parser
    
    res=chain.invoke(input={"information":linkedin_data})
    print(res)


if __name__=="__main__":
    print("hello iceberg")
    icebreak(name="dhruv bhagat TapTalent.ai")