"""
           User
            │
            ▼
   Question + Chat History
            │
            ▼
  History Aware Retriever
            │
            ▼
       Vector Store
   (Flipkart Reviews)
            │
            ▼
      Retrieved Docs
            │
            ▼
      QA Prompt + LLM
            │
            ▼
          Answer
            │
            ▼
      Save to Memory
"""

from langchain_groq import ChatGroq # generate the convo reponses by llm
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

from langchain_core.runnables.history import RunnableWithMessageHistory
# these 2 resposiblie for message history 
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from src.config import Config


class RAGChainBuilder:
    def __init__(self,vector_store):
        self.vector_store = vector_store
        self.model = ChatGroq(model=Config.RAG_MODEL,temperature=0.5)
        self.history_store={}
      
      # the memory manager -> if i never seen this user create a new notebook for them sle rreturn their existing one 
    def _get_history(self,session_id:str) -> BaseChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]
    
    # the core logic where it connects 5 pieces [ 1-retriver,2-context promt(rephrasing step)
    #3-QA PROMT (answering step, 4- chaining everything togther, 5- wrapping with memory]
    def build_chain(self):
        
        retriever = self.vector_store.as_retriever(search_kwargs={'k':3})

        '''
        context_prompt = ChatPromptTemplate.from_messages([
        ("system", "...rewrite it as a standalone question."),
        MessagesPlaceholder(variable_name="chat_history"),  # injects past messages here
        ("human", "{input}")   # current user question])
        Problem it solves: If user asks "What about its battery?" — the retriever won't know what "its" means.
        Solution: This prompt tells the AI to rephrase it as "What is the battery life of the Samsung Galaxy S23?" using chat history.
        
        '''
        
        context_promt = ChatPromptTemplate.from_messages([('system','Given the chat history and the user question, rewrite it as a standalone question.'),
                                            MessagesPlaceholder(variable_name='chat_history'),
                                            ('human','{input}')
                                            ])
        
        qa_promt =ChatPromptTemplate.from_messages([
            ('system',""" You're an e-commerce bot answering product-related queries using reviews and titles.
                          Stick to context. Be concise and helpful.\n\nCONTEXT:\n{context}\n\nQUESTION: {input}"""),  # {context} = retrieved products
                          MessagesPlaceholder(variable_name='chat_history'),
                          ('human','{input}') ])
        # step A : Takes question + history -> produces a better standalone question ->retrives doc
        history_aware_retriever = create_history_aware_retriever(self.model,retriever,context_promt)
        # Step B : takes retrived docs + question -> produces final answer
        question_answer_chain = create_stuff_documents_chain(self.model,qa_promt)
        # STEP A feeds into STEP B
        rag_chain= create_retrieval_chain(history_aware_retriever,question_answer_chain)

        return RunnableWithMessageHistory(
            rag_chain,
            self._get_history,  #tells it how to get/store history
            input_messages_key='input',  # user question ges here
            history_messages_key='chat_history', # history gets injected here
            output_messages_key='answer' #final answe comes out here

        )

      
        
"""
```
This wrapper **automatically saves** each conversation turn to `history_store`.

---

## Full Flow Visualized
```
User: "What about its battery?" (session: "user_123")
          │
          ▼
   RunnableWithMessageHistory
   loads history for "user_123": ["Tell me about Galaxy S23"]
          │
          ▼
   context_prompt + model
   → rephrases to: "What is the battery life of Galaxy S23?"
          │
          ▼
   retriever searches vector_store
   → returns 3 relevant product/review documents
          │
          ▼
   qa_prompt + model
   → "The Galaxy S23 has a 3900mAh battery, lasting ~24hrs per reviews."
          │
          ▼
   history_store["user_123"] updated with this new Q&A
The key insight is that {input}, {context}, and {chat_history} are placeholder variables that get filled in automatically as data flows through the pipeline.
  
"""