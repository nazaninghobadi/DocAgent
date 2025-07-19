from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

class LLMProvider:
    """
    Provides access to a chat language model via OpenRouter API.
    This model is tailored for student/academic support use cases.
    """

    def __init__(self, api_key: str, model_name: str = "mistralai/mistral-7b-instruct"):
        self.api_key = api_key
        self.model_name = model_name

    def get_chat_model(self) -> BaseChatModel:
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            model=self.model_name
        )
