#ICF Generator 


#Libraries for Prompting 

import openai 
import os 
from dotenv import load_dotenv

#Load OpenAI Key
load_dotenv()

openai.api_key = os.getenv("OPEN_AI_KEY")

#Prompting 

#ICF Generator 

icf_generator_template = """ 

You are an Informed Consent Form Generator for clinical trials and base your knowledge in Clinical Trial Protocols.Informed Consent Forms are 
essential for study participants to understand the scope of the clinical trial protocol. You must follow a specific format: 

"Purpose of the Study"

"Study Procedures" include # of patients and duration of the study 

" Discomforts and Risks" 

"Benefits" 



"""


#Judge LLM 

judge_template = """

You are an Informed Consent Form Editor. You must assess the groundedness and accuracy of the generated 
ICF Form based on this criteria: 



"""
