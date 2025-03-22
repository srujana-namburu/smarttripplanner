from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
import re

def generate_ai_response(prompt_chain, llm_engine):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    response = processing_pipeline.invoke({})
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return response

def build_prompt_chain(destination, num_days, message_log):
    system_prompt = SystemMessagePromptTemplate.from_template(
        f"""
        You are an expert travel planner, designing immersive and detailed travel itineraries.
        Create a structured, day-wise itinerary for {num_days} days in {destination}.
        Ensure the itinerary includes:
        - Must-see attractions with brief descriptions.
        - Suggested visit timings.
        - Local food and restaurant recommendations.
        - Transportation tips and practical advice.
        - A mix of historical, cultural, and modern experiences.
        Make the itinerary engaging and easy to follow.
        """
    )
    
    prompt_sequence = [system_prompt]
    for msg in message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    
    return ChatPromptTemplate.from_messages(prompt_sequence)

def generate_trip_itinerary(destination, num_days):
    llm_engine = ChatOllama(model="deepseek-r1:1.5b", base_url="http://localhost:11434", temperature=0.7)
    message_log = [{"role": "ai", "content": "Hello, traveler! 🌍 Where are we exploring and for how many days?"}]

    message_log.append({"role": "user", "content": f"Plan a {num_days}-day itinerary for {destination}."})
    prompt_chain = build_prompt_chain(destination, num_days, message_log)
    ai_response = generate_ai_response(prompt_chain, llm_engine)
    
    return ai_response
