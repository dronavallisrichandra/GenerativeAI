
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import CohereEmbeddings

pdf_Loader = PyPDFDirectoryLoader ("./pdf-docs")
pages_dir = pdf_Loader.load()
Loaders = [pdf_Loader]
documents = []
for loader in loaders:
    documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=100)
all_documents = text_splitter.split_documents(documents)
print(f"Total number of documents: {len(all_documents)}")

#setup OCI Generative AI LLM

embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-english-v3.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaaf4a2mihhz26covfccz5yqnm2c4fr6qgwxoexfu7xadgf5h4zsvoq",
    model_kwargs={"truncate" :True}

)



#Ster 2 - since OCIGenAlEmbeddings accepts only 96 documents in one run,we will input documents in batches.
# Set the batch size
batch_size = 96
# Calculate the number of batches
num_batches = len(all_documents) // batch_size + (len(all_documents) % batch_size > 0)
db = FAISS.from_texts(texts,embeddings)
retv = db.as_retriever()
# Iterate over batches
for batch_num in range(num_batches):
    # Calculate start and end indices for the current batch
    start_index = batch_num * batch_size
    end_index = (batch_num + 1) * batch_size
    # Extrac documents for the current batch
    batch_documents = all_documents[start_index:end_index]
    # your code to process each document goes here
    retv.add_documents(batch_documents)
    print(start_index,end_index)

#here we persist the collection
db.save_local("faiss_index")


