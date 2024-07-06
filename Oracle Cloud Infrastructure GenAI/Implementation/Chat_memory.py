from langchain.memory.buffer import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
from langchain.chains import LLMChain
from langchain.prompts import ( ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, )
from langchain_community.llms import OCIGenAI
# use default authN method API-key
llm = OCIGenAI(
    model_id="cohere.command-light",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaaf4a2mihhz26covfccz5yqnm2c4fr6qgwxoexfu7xadgf5h4zsvoq",
    model_kwangs={"max_tokens":100}
)
#here we create a prompt

prompt = ChatPromptTemplate(
messages=[
SystemMessagePromptTemplate.from_template(
    "You are a nice chatbot who explain in steps."
    ),
    HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
summary_memory = ConversationSummaryMemory (llm=llm, memory_key="chat_history")
#Step.here.we.cceate.a.conversation.chain.using,Lmm. prompt.and,memory.
conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)

conversation.invoke({"question": "What is the capital of India"})

#here we print all the messages in memory
print(memory.chat_memory.messages)
print(summary_memory.chat_memory.messages)
print("Summary of the conversation is --> " +summary_memory.buffer)


# we will again ask one more question and see the memory of message by printing

conversation.invoke({"question": "What is the capital of Andhra Pradesh"})

#here we print all the messages in memory
print(memory.chat_memory.messages)
print(summary_memory.chat_memory.messages)
print("Summary of the conversation is --> " +summary_memory.buffer)