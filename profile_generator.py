from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Read the variables
openai_api_key = os.environ.get('OPENAI_API_KEY')
langchain_tracing_v2 = os.environ.get('LANGCHAIN_TRACING_V2')
langchain_endpoint = os.environ.get('LANGCHAIN_ENDPOINT')
langchain_api_key = os.environ.get('LANGCHAIN_API_KEY')
langchain_project = os.environ.get('LANGCHAIN_PROJECT')
pinecone_api_key = os.environ.get('PINECONE_API_KEY')
pinecone_env = os.environ.get('PINECONE_ENV')


import json
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def gather_responses():
    # Defining the questions
    questions = {
        'organization_overview': "Provide a brief description of your organization's primary focus and its major accomplishments in the past year.",
        'beneficiary_demographics': "Describe the main demographics of the individuals or communities your organization serves. Include age groups, gender, and any other relevant details.",
        'current_projects': "Briefly describe any ongoing projects or initiatives your organization is currently involved in.",
        'previous_collaborations': "Mention any significant collaborations, partnerships, or joint projects your organization has had in the past.",
        'operational_capabilities': "Describe the operational strengths of your organization. What can you execute exceptionally well?"
    }
    
    # Gathering responses and storing them as Q&A pairs
    qa_pairs = []
    for key, question in questions.items():
        print(question)
        answer = input("Please provide your answer: ")  # This line allows for manual input
        qa_pairs.append({
            "question": question,
            "answer": answer
        })

    # Convert Q&A pairs list to JSON-formatted string
    json_string = json.dumps(qa_pairs, ensure_ascii=False, indent=4)
    
    return json_string

def generate_output_profile(qa_string):
    prompt_template = """You are an AI assistant working in social area. 
    Your mission is to carefully evaluate "question - answer" pairs saved in a python dictionary {qa_string}.
    Provide a well-structured, detailed description of a non-profit profile."""

    llm = OpenAI(temperature=0)
    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template)
    )
    return llm_chain({"qa_string": qa_string})

if __name__ == "__main__":
    # Run the function to gather and store Q&A pairs
    qa_string = gather_responses()
    print("\nStored Q&A Pairs in JSON format:")
    print(qa_string)
    output_profile = generate_output_profile(qa_string)
    print("\nGenerated Non-Profit Profile:")
    print(output_profile)
