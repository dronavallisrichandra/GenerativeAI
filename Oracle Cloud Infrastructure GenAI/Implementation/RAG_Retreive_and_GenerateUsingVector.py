from langchain.chains import RetrievalQA
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OCIGenAI
from langchain_community.embeddings import OCIGenAIEmbeddings

llm = OCIGenAI(
    model_id="cohere.command-light",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaaf4a2mihhz26covfccz5yqnm2c4fr6qgwxoexfu7xadgf5h4zsvoq",
    model_kwangs={"max_tokens":100}
)

# here we connet to a chromadb server , we need to run the chromadb server before we connect to it.
client = chromadb.HttpClient(host="127.0.0.1")

# here we create embeddings using 'cohere.embed-english-light-v2.0" model

embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-english-v3.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaaf4a2mihhz26covfccz5yqnm2c4fr6qgwxoexfu7xadgf5h4zsvoq",
    model_kwargs={"truncate" :True}
)

# here we create a retrirver that gets relevant documents ( similar in meaning to a query )

db = Chroma(client=client, embedding_function=embeddings)

retv = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# here we can explore how similar documents to the query are returned by printing the document metadata, This step is optional

docs = retv.get_relevant_documents("Tell us which module is most relevant to LLMs and Generative AI")

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

pretty_print_docs(docs)

for doc in docs:
    print(doc.metadata)

# here we create a retreival chain that takes llm, retreiver objects and invoke it to get a response

Chain = RetrievalQA.from_chain_type(llm=llm, retriever=retv, return_source_documents=True)

response = chain.invoke("Tell us which module is relevant to LLMs and Generative AI")
print(response)