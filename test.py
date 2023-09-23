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

import os
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA

embeddings = OpenAIEmbeddings()

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  
    environment=os.getenv("PINECONE_ENV"),  
)

index_name = "test"

# Load the existing index
vectorstore = Pinecone.from_existing_index(index_name, embeddings)
retriever = vectorstore.as_retriever()

question = "Are there any projects that might help in training professionals to better care for the elderly?"

template = """You are helpful and very polite AI assistant non-profits.

Your goal is to help non-profits seeking funding opportunities by answering their questions using provided context below as your knowledge base.

Remember that a lot of non-profits count on you to find grant oportunities that match their objectives.

If you don't know the answer, just say that you don't know.  
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever(),chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

result = qa_chain({"query": question})
print(result["result"])