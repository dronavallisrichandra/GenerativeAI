from langchain_community.llms import OCIGenAI


endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

llm = OCIGenAI(
    model_id="cohere.command",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.user.oc1..aaaaaaaalmyydxz3ysp2agod6dl4x62hns5yqhhv2dpaj2ixn7s4mp7uk7xq",
    model_kwargs={"max_tokens":200}
)



#invoke llm with a fixed text input
response = llm.invoke("tell me one fact about space", temperature=0.7)

print("Case1 Response - >" + response)

# using string prompt to accept text input, here we creat a template and declare a input variable {human input}

template = """ you are a chatbot having a conversation with a human.
Human: {human_input} + {city}:"""

#step 4 - here we create a prompt using the template

prompt = PromptTemplate(input_variables=["human_input","city"], template=template)

prompt_val = prompt.invoke({"human_input":"tell us in a exciting tone about", "city":"Los vegas"})

print("Prompt String is ->")
print(prompt_val.to_string())

#here we declare a chain that begins with a prompt, next llm and finally output parser

chain = prompt | llm

#invoke a chain and provide input question

response = chain.invoke({"human_input":"tell us in a exciting tone about", "city":"Las Vegas"})

#print the prompt response

print("Case2 Response" + response)



###  chat prompt to accept text input, here we create a chat template and use HumanMessage and SystemMessage

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are a chatbot that explains in steps."),
        ("ai","I shall explain in steps"),
        ("human","{input}"),
    ]
)

chain = prompt | llm

response = chain.invoke({"input":"tell us in a exciting tone about?"})

#print the prompt response

print("Case3 Response" + response)


## use chat message template and message and pass {question}

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are a very knowledgeable scientist who provides accurate and eloquent answers to scientific questions."),
        ("human","{question}"),
    ]
)

#create a chain using LLMChain class and invoke a chain to get a response
#legacy_chain

chain = LLMChain(llm=llm, prompt=prompt, output_parser=StrOutputParser())
response = chain.invoke({"question":"what are basic elements of a matter"})
print(response)

#lecl chain

runnable = prompt | llm | StrOutputParser()
response = runnable.invoke({"question": "What are basic elements of a matter"})
print(response)
