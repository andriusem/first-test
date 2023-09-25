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

def gather_responses():
    # Defining the questions
    questions = {
        'organization_overview': "Provide a brief description of your organization's primary focus and its major accomplishments in the past year.",
        'operational_geographies': "List the primary regions or areas where your organization operates. Include countries, states, or cities if possible.",
        'beneficiary_demographics': "Describe the main demographics of the individuals or communities your organization serves. Include age groups, gender, and any other relevant details.",
        'current_projects': "Briefly describe any ongoing projects or initiatives your organization is currently involved in.",
        'funding_range': "Specify the range of funding you are open to receiving. Provide a minimum and maximum value.",
        'duration_flexibility': "For how long can your organization commit to a project or initiative related to a grant? Specify a range in months or years.",
        'previous_collaborations': "Mention any significant collaborations, partnerships, or joint projects your organization has had in the past.",
        'adaptability': "On a scale of 1 to 10, how flexible is your organization in adapting to the requirements of a new grant or project?",
        'resource_needs': "Apart from financial aid, what other types of support or resources does your organization frequently require?",
        'operational_capabilities': "Describe the operational strengths of your organization. What can you execute exceptionally well?"
    }
    
    # Fictional answers for Silver Horizons
    answers = {
        'organization_overview': "Silver Horizons is primarily dedicated to providing comprehensive care and support for the elderly, focusing on combating loneliness, mental health improvement, and overall well-being. In the past year, we successfully launched our 'ElderBuddy' program, which paired over 2,000 seniors with volunteers for weekly companionship and assistance.",
        'operational_geographies': "We predominantly operate in urban areas of the East Coast, with a significant presence in cities like New York, Boston, and Philadelphia. Additionally, we have smaller initiatives running in cities such as Baltimore and Newark.",
        'beneficiary_demographics': "Our primary beneficiaries are elderly individuals aged 60 and above. We also have specialized programs catering to women aged 30-50 who have faced distressing situations. Lastly, our prevention programs target youths and young adults between 15-30 years old in urban regions.",
        'current_projects': "Apart from our ongoing 'ElderBuddy' program, we have initiated 'EmpowerHER,' a support group and resource center for women in distress. We're also piloting 'CityClean,' aimed at preventing substance abuse in young adults within urban locales.",
        'funding_range': "We are flexible and open to grants ranging from $50,000 to $500,000, depending on the scope and reach of the proposed project.",
        'duration_flexibility': "We can commit to projects lasting from 6 months up to 3 years, based on the specific needs and objectives of the initiative.",
        'previous_collaborations': "In the past, we've collaborated with organizations such as 'UrbanYouth Rise,' 'Seniors First Foundation,' and 'Women's Beacon' to augment our reach and impact.",
        'adaptability': "8",
        'resource_needs': "Beyond financial support, we often require expertise in mental health counseling, training for our volunteers, technological tools for remote assistance, and awareness campaign materials for our prevention programs.",
        'operational_capabilities': "Silver Horizons excels in community outreach and program execution in urban areas. Our strength lies in our vast network of trained volunteers, our collaborations with local health centers, and our ability to quickly mobilize resources for emergent community needs."
        }
    
    # Gathering responses and storing them as Q&A pairs
    qa_pairs = []
    for key, question in questions.items():
        print(question)
        # Use predefined answers for the fictional organization
        answer = answers[key]
        print(f"Answer: {answer}\n")  # This will print the answer. Remove this line if you want to manually input answers.
        qa_pairs.append({
            "question": question,
            "answer": answer
        })

    # Convert Q&A pairs list to JSON-formatted string
    json_string = json.dumps(qa_pairs, ensure_ascii=False, indent=4)
    
    return json_string

# Run the function to gather and store Q&A pairs
qa_string = gather_responses()
print("\nStored Q&A Pairs in JSON format:")
print(qa_string)

prompt_template = """You are an AI assistant working in social area. 
You mission is to carefuly evaluate "question - answer" pairs saved in python dictionary {qa_string}.
Provide a well-structured, detailed description of a non-profit profile."""

llm = OpenAI(temperature=0)
llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)
print(llm_chain({"qa_string": qa_string}))