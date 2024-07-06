import os
from uuid import uuid4
from langsmith import Client

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = " "


#create dataset for evaluation

dataset_inputs = [
    "tell us about OCIGen AI Cource",
    "tell us about deep learning",
    "tell us about a modeule relevant to LLMs and Transformers OCIGen AI Cource"
]

dataset_outputs = [
    {"must_mention": ["AI", "LLM"]},
    {"must_mention": ["CNN", "Neural Network"]},
    {"must_mention": ["Module 5", "Transformer", "LLM"]}

]


client = Client()
dataset_name = "AIFoundations-111"

dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="AI Foundations QA.",
)

client.create_examples(
    inputs=[{"question": q} for q in dataset_inputs],
    outputs=dataset_outputs,
    dataset_id=dataset.id,
)