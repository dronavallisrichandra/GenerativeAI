import os
from uuid import uuid4
import langsmith
from langchain import smith
from langchain.smith import RunEvalConfig

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain.chains import RetrievalQA

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = " "


from langchain_community.llms import OCIGenAI

llm = OCIGenAI(
    model_id="cohere.command",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaaf4a2mihhz26covfccz5yqnm2c4fr6qgwxoexfu7xadgf5h4zsvoq",
    model_kwangs={"max_tokens":100}
)


embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-english-v3.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaaf4a2mihhz26covfccz5yqnm2c4fr6qgwxoexfu7xadgf5h4zsvoq",
    model_kwargs={"truncate" :True}
)

db = FAISS.load_local("faiss_index",embeddings)

retv = db.as_retriever(search_kwargs={"k": 8})

chain = RetrievalQA.from_chain_type(llm=llm, retriever=retv)

# define the evaluators to apply
# default criteris are implemented for the following aspects: conciseness, relevance,
# correctness, coherence, harmfulness, maliciousness, helpfulness, controversiality, misogyny, and criminality

eval_config = smith.RunEvalConfig(
    evaluators=[
        "cot_qa",
        RunEvalConfig.Criteria("relevance")
    ],
    custom_evaluators = [],
    eval_llm=llm
)

client = langsmith.Client()

chain_results = client.run_on_dataset(
    dataset_name="AIFoundations-111",
    llm_or_chain_factory=chain,
    evaluation=eval_config,
    concurrency_level=5,
    verbose=True,
)