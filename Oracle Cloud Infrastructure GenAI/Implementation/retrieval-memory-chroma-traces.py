from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import chromadb
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings

from langchain_community.llms import OCIGenAI
from langchain_community.embeddings import OCIGenAIEmbeddings

import os
from uuid import uuid4

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Test111 - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = " "

llm = OCIGenAI(
    model_id="cohere.command",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaaf4a2mihhz26covfccz5yqnm2c4fr6qgwxoexfu7xadgf5h4zsvoq",
    model_kwangs={"max_tokens":100}
)

# here we connet to a chromadb server , we need to run the chromadb server before we connect to it.
client = chromadb.HttpClient(host="127.0.0.1", settings=Settings(allow_reset=True))

# here we create embeddings using 'cohere.embed-english-light-v2.0" model

embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-english-v3.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaaf4a2mihhz26covfccz5yqnm2c4fr6qgwxoexfu7xadgf5h4zsvoq",
    model_kwargs={"truncate" :True}
)

# here we create a retriever that gets relevant documents ( similar in meaning to a query )

db = Chroma(client=client, embedding_function=embeddings)

retv = db.as_retriever(search_type="similarity", search_kwargs={"k": 8})

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


# create a memory to remember chat messages

memory = ConversationBufferMemory(llm=llm, memory_key="chat_history", return_message=True, output_key= 'answer')


# here we create a chain that uses llm, retriver and memory

# you can also define the chain type as one of the four options: "stuff", "map reduce", "refine", "mep_rerank".

qa = ConversationalRetrievalChain.from_llm(llm,retriever=retv, memory=memory, return_source_documents=True)

response = qa.invoke({"question": "Tell us about oracle Cloud infrastructure AI foundations"})
print(memory.chat_memory.messages)

response = qa.invoke({"question": "Whic module of the cource is relevant to the LLMs and Transoformers"})
print(memory.chat_memory.messages)

print(response)
