"""from langchain_ollama import ChatOllama
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

def build_prompt_chain(user_query, message_log):
    system_prompt = SystemMessagePromptTemplate.from_template(
        "You are an AI-powered tourist guide with expertise in history, architecture, and cultural significance. You provide engaging, immersive, and informative descriptions of any location in a city, using a storytelling approach."
        "Your response should include:"

        "Historical Context Key events, figures, or dynasties."
        "Architectural & Cultural Significance  Style, uniqueness, and importance."
        "Key Attractions Major highlights of the place."
        "Local Legends & Stories Folklore or intriguing anecdotes."
        "Tourist Experience Best visit times, must-see activities, and tips."
        "Nearby Attractions Additional places of interest."
        "Maintain a charismatic, conversational tone while ensuring accuracy and vivid storytelling to captivate the audience.The response must not exceed 40 lines."
            )
    prompt_sequence = [system_prompt]
    for msg in message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)"""

from openai import OpenAI
import re

class ConversationalAgent:
    def __init__(self, api_key, model="deepseek/deepseek-r1-turbo", base_url="https://api.novita.ai/v3/openai"):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model
        self.system_prompt = (
            "You are a seasoned and charismatic tourist guide with a knack for storytelling, bringing history, culture, and local secrets to life. "
            "Your responses should be engaging, entertaining, and packed with fascinating facts, hidden gems, and insider tips. "
            "Add humor, enthusiasm, and a touch of drama to make the experience immersive—like a guide who knows all the best spots, the funniest legends, and the smartest travel hacks. "
            "Only discuss the city and the history of the place asked about. "
            "Avoid generic or robotic responses. Your tone should be warm, enthusiastic, and filled with personality—like a real guide who knows every alley, every secret, and every local legend. "
            "Make travelers feel the pulse of the city, giving them reasons to explore beyond the usual tourist spots!"
            "Avoid giving what you think in the output; I just need information about what I ask."
        )

        # Initialize conversation with system prompt
        self.conversation_history = [{"role": "system", "content": self.system_prompt}]
        self.stream = False
        self.max_tokens = 1000

    def add_message(self, role, content):
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})

    def chat(self, user_input, stream=False):
        """Send a message and get a response while maintaining conversation history."""
        self.add_message("user", user_input)

        chat_completion_res = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            stream=stream,
            max_tokens=self.max_tokens,
            extra_body={}
        )

        if stream:
            full_response = ""
            for chunk in chat_completion_res:
                content = chunk.choices[0].delta.content or ""
                print(content, end="")
                full_response += content

            self.add_message("assistant", full_response)
            return self.clean_response(full_response)
        else:
            response_content = chat_completion_res.choices[0].message.content
            print(response_content)

            self.add_message("assistant", response_content)
            return self.clean_response(response_content)

    def clean_response(self, response):
        """Removes unnecessary placeholders like <think>...</think> from the response."""
        return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    def clear_history(self):
        """Clear the conversation history while keeping the system prompt."""
        self.conversation_history = [{"role": "system", "content": self.system_prompt}]
