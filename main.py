import os
import json
import pinecone
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import LLMChain

# Load the .env file
load_dotenv()

# Read the environment variables
openai_api_key = os.environ.get('OPENAI_API_KEY')
langchain_tracing_v2 = os.environ.get('LANGCHAIN_TRACING_V2')
langchain_endpoint = os.environ.get('LANGCHAIN_ENDPOINT')
langchain_api_key = os.environ.get('LANGCHAIN_API_KEY')
langchain_project = os.environ.get('LANGCHAIN_PROJECT')
pinecone_api_key = os.environ.get('PINECONE_API_KEY')
pinecone_env = os.environ.get('PINECONE_ENV')

def read_output_profile_from_file():
    with open("output_profile.txt", "r") as file:
        data = json.load(file)
        return data.get("text", "")

question = read_output_profile_from_file()
embeddings = OpenAIEmbeddings()

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)

index_name = "test"

vectorstore = Pinecone.from_existing_index(index_name, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={'k': 10})

# Get the 10 most similar documents from Pinecone
# similar_docs = vectorstore.similarity_search(
#     question,  # our search query
#     k=10  # return 10 most relevant docs
# )

# You can further process the similar_docs if needed before proceeding.

template = """You are a powerful AI system with perfect reasoning and planning skills.

Your goal is to analyse the profile of a non-profit organization and find the most relevant grant opportunities, provided as a context below, for it.

Remember that a lot of non-profits count on you to find grant opportunities that match their profile.

Provide the name of the grant and other relevant details.
Important! Only use the provided grants as a context. Do not invent information about the grants.
Question: 
{question}
Helpful Answer:{context}
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

llm = ChatOpenAI(model_name="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

result = qa_chain({"query": question})
print(result["result"])
