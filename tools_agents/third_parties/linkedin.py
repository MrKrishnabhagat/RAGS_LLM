import os
import requests
from dotenv import load_dotenv
load_dotenv()


def scrape_linkedin_profile(linkedin_url:str,mock:bool=False):
    """scrape information from linkedin profiles,
    manually scrape information from linkedin profile"""
    if mock:  
        linkedin_profile_url="https://gist.githubusercontent.com/emarco177/0d6a3f93dd06634d95e46a2782ed7490/raw/fad4d7a87e3e934ad52ba2a968bad9eb45128665/eden-marco.json"
        response=requests.get(linkedin_profile_url,timeout=10)
    else:
        api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
        header_dic = {'Authorization': f'Bearer {os.environ.get("PROXYXURL_API_KEY")}'}
        response = requests.get(api_endpoint,
                                params={"url":linkedin_url},
                                headers=header_dic,
                                timeout=30)    
    data=response.json()
    return data    

if __name__=="__main__":
    print(
        scrape_linkedin_profile(linkedin_url="https://www.linkedin.com/in/krishna-bhagat-72744b281/")
    )