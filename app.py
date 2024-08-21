from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import uvicorn

app = FastAPI()

class UserQuery(BaseModel):
    question: str

groq_api_key = "------>"

# Model configuration
model = 'llama3-8b-8192'
conversational_memory_length = 5
system_prompt = "You are a friendly and knowledgeable chatbot."

memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

groq_chat = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=model
)

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}"),
    ]
)

conversation = LLMChain(
    llm=groq_chat,
    prompt=prompt,
    verbose=True,
    memory=memory,
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to  chatbot API!"}

@app.post("/chat")
async def chat(user_query: UserQuery):
    user_question = user_query.question
    if not user_question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    response = conversation.predict(human_input=user_question)
    message = {'human': user_question, 'AI': response}
    memory.save_context({'input': user_question}, {'output': response})

    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
