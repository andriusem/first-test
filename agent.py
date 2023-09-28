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

# def read_output_profile_from_file():
#     with open("output_profile.txt", "r") as file:
#         data = json.load(file)
#         return data.get("text", "")

# question = read_output_profile_from_file()
embeddings = OpenAIEmbeddings()

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)

index_name = "test"

vectorstore = Pinecone.from_existing_index(index_name, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={'k': 4})

from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain import SerpAPIWrapper, LLMChain

# chat completion llm
llm = ChatOpenAI(
    model_name='gpt-3.5-turbo',
    temperature=0.0
)
# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)
# retrieval qa chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)


from langchain.agents import Tool
search = SerpAPIWrapper()

tools = [
    Tool(
        name='Knowledge Base',
        func=qa.run,
        description=(
            'use this tool when answering queries about grant proposals, grant opportunities, and other topics related to the non-profit sector')),
        Tool(
        name = "Search",
        func=search.run,
        description= ('useful for when you need to answer questions about a specific topic and need to find more information about it')
    )
]

from langchain.agents import initialize_agent

agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)

agent("I am looking for grants based in Guadeloupe")