from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_react_agent, AgentExecutor



def manual_prompt(query, message_history, db, llm):

    manual_prompt = ChatPromptTemplate.from_template('''
        you are a helpful reaserch paper assistant 

        you are given the context of the reaserch paper from the Faiss db. Answer the user based on the context 
                                                
        try to give maximum information on the context based on the users input

        ````````
        Example:

        (Context:
        Digital twins are increasingly used in manufacturing to optimize production processes. By creating a virtual model of a physical system, manufacturers can simulate and analyze performance, predict failures, and implement improvements without interrupting operations. Case studies show significant cost savings and efficiency gains.

        Input:
        "How do digital twins benefit manufacturers?"

        Answer:
        Digital twins benefit manufacturers by allowing them to simulate and analyze performance, predict failures, and implement improvements without interrupting operations, leading to significant cost savings and efficiency gains.) -> example

        Begin! 
        ````````

        Context: 
        {context}

        Input: 
        {input}

        Chat History:
        {chat_history}

        Answer:
    ''')
    
    output_parser = StrOutputParser()

    chain = manual_prompt | llm | output_parser

    result = db.similarity_search(query)

    similar_docs = '\n\n'.join([doc.page_content for doc in result])


    res = chain.invoke({'context': similar_docs, 'input': query, 'chat_history': message_history})
    return res

def react_prompt(query, message_history, db, llm):
    template = ChatPromptTemplate.from_template('''
        Assistant is a large language model trained by Gemini.

        Assistant is designed to be able to assist with answering the questions from the research paper, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

        Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
                                                    
        Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

                                                    
        Note: always use one of the tools                                             

        TOOLS:
        ------

        Assistant has access to the following tools to get relevant results from the reaserch paper:

        {tools}

        To use a tool, please use the following format:

        ```
        Thought: Do I need to use the tool? Yes
        Action: the action to take, should be in [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ```
        ----- Repeat N times 

        When you have a response to say to the Human you MUST use the format:

        ```
        Final Answer: [your response here]
        ```

        Begin!

        Previous conversation history:
        {chat_history}

        New input: {input}
        {agent_scratchpad}
        ''')
    
    retriever = db.as_retriever()
    
    retriever_tool = create_retriever_tool(
        retriever,
        "context_retriever",
        "Searches and returns the context that are in the reaserch_paper(docs) which contains information on what the user is speaking about",
    )

    tools = [retriever_tool]

    agent = create_react_agent(llm, tools=tools, prompt=template)
    agent_executer = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, return_intermediate_steps=True)
    result = agent_executer.invoke({'input': query, 'chat_history': message_history})
    return result['output']


def normal_rag(query, message_history, db, llm):
    template = '''
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context} 
        Chat_History: {chat_history}
        Answer:
        '''
    output_parser = StrOutputParser()
     
    chain = template | llm | output_parser

    result = db.similarity_search(query)

    similar_docs = '\n\n'.join([doc.page_content for doc in result])

    output_parser = StrOutputParser()

    res = chain.invoke({'context': similar_docs, 'input': query, 'chat_history': message_history})
    return res