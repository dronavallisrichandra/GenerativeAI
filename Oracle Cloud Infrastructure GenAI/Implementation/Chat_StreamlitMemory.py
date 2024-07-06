from langchain.memory.buffer import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OCIGenAI
# use default authN method API-key
llm = OCIGenAI(
    model_id="cohere.command-light",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaaf4a2mihhz26covfccz5yqnm2c4fr6qgwxoexfu7xadgf5h4zsvoq",
    model_kwangs={"max_tokens":100}
)
#here we create a history with a key chat messages

#StreamlitChatMessageHistory will store messages in Streamlit session state at the specified key=

history = StreamlitChatMessageHistory(Key="Chat_messages")

#here we create a memory object

memory = ConversationBufferMemory(chat_memory=history)

#here we create template and prompt to accept a question

template = """ you are a chatbot having a conversation with a human.
Human: {human_input} + AI: """

prompt = PromptTemplate(input_variables=["human_input"], template=template)

# here we create a chain object

llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

#here we use streamlit to print all messages in the memory, create text input, run chain and the question and response is automatically put in the StreamlitChatMessageHistory

import streamlit as st

st.title(" Welcome to the Chatbot")
for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

if x := st.chat_input():
    st.chat_message("human").write(x)

    # As usual, new messages are added to history when the chain is called

    response = llm_chain.run(x)
    st.chat_message("ai").write(response)




