from typing import List, Dict, Any, Optional
import logging
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.tools import BaseTool

from modules.llm_provider import LLMProvider
from modules.tools import (
    search_knowledge,
    search_knowledge_with_scores,
    create_summarize_tool,
    get_vector_store_stats,
    model_manager
)

logger = logging.getLogger(__name__)


class SmartStudyAgent:
    """
    Professional AI agent for document interaction and study assistance.
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "mistralai/mistral-7b-instruct",
        max_iterations: int = 5,
        verbose: bool = True
    ):
        """
        Initialize the Smart Study Agent.
        
        Args:
            api_key: OpenRouter API key
            model_name: LLM model name
            max_iterations: Maximum agent iterations
            verbose: Enable verbose logging
        """
        self.api_key = api_key
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.agent_executor: Optional[AgentExecutor] = None
        
        # Initialize components
        self._initialize_agent()
    
    def _initialize_agent(self) -> None:
        """Initialize the agent with tools and LLM."""
        try:
            llm_provider = model_manager.get_llm(self.api_key, self.model_name)
            llm = llm_provider.get_chat_model()
            
            tools = self._setup_tools()
            
            prompt = self._create_prompt()
            
            agent = create_react_agent(
                llm=llm,
                tools=tools,
                prompt=prompt
            )
            
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=self.verbose,
                max_iterations=self.max_iterations,
                handle_parsing_errors=True,
                return_intermediate_steps=True
            )
            
            logger.info("Smart Study Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Agent initialization failed: {str(e)}")
            raise RuntimeError(f"Failed to initialize agent: {str(e)}")
    
    def _setup_tools(self) -> List[BaseTool]:
        """Setup and configure agent tools."""
        tools = [
            search_knowledge,
            search_knowledge_with_scores,
            get_vector_store_stats,
            create_summarize_tool(self.api_key, self.model_name)
        ]
        
        logger.info(f"Configured {len(tools)} tools for agent")
        return tools
    
    def _create_prompt(self) -> PromptTemplate:
        """Create custom prompt template for the agent."""
        template = """
You are a Smart Study Buddy - an AI assistant specialized in helping students with their study materials.

Your capabilities include:
1. **search_knowledge**: Search through uploaded documents using semantic similarity
2. **search_knowledge_with_scores**: Search with similarity confidence scores
3. **summarize_knowledge**: Create summaries of relevant document sections
4. **get_vector_store_stats**: Get information about the document database

Guidelines for helping students:
- Always be clear and educational in your explanations
- When searching documents, use relevant keywords from the student's question
- Provide specific examples and quotes when possible
- If information is not found, suggest alternative search terms
- Break down complex topics into digestible parts
- Encourage active learning by asking follow-up questions

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""
        
        return PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in self._setup_tools()]),
                "tool_names": ", ".join([tool.name for tool in self._setup_tools()])
            }
        )
    
    def chat(self, query: str, return_steps: bool = False) -> Dict[str, Any]:
        """
        Process a chat query with the agent.
        
        Args:
            query: User's question or request
            return_steps: Whether to return intermediate steps
        
        Returns:
            Dictionary containing response and optional intermediate steps
        """
        if not query.strip():
            return {"error": "Empty query provided"}
        
        if not self.agent_executor:
            return {"error": "Agent not initialized"}
        
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            # Execute agent
            result = self.agent_executor.invoke(
                {"input": query},
                return_only_outputs=not return_steps
            )
            
            response = {
                "answer": result.get("output", "No answer generated"),
                "success": True
            }
            
            if return_steps:
                response["intermediate_steps"] = result.get("intermediate_steps", [])
            
            logger.info("Query processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Chat processing error: {str(e)}")
            return {
                "error": f"Failed to process query: {str(e)}",
                "success": False
            }
    
    def batch_chat(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of user queries
        
        Returns:
            List of response dictionaries
        """
        results = []
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing batch query {i}/{len(queries)}")
            result = self.chat(query)
            results.append(result)
        
        return results
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about the agent configuration.
        
        Returns:
            Dictionary containing agent information
        """
        return {
            "model_name": self.model_name,
            "max_iterations": self.max_iterations,
            "verbose": self.verbose,
            "tools_count": len(self._setup_tools()),
            "tool_names": [tool.name for tool in self._setup_tools()],
            "is_initialized": self.agent_executor is not None
        }
    
    def clear_cache(self) -> None:
        """Clear all cached models and data."""
        model_manager.clear_cache()
        logger.info("Agent cache cleared")


def build_agent(api_key: str) -> SmartStudyAgent:
    """
    Build a Smart Study Agent with default settings.
    
    Args:
        api_key: OpenRouter API key
    
    Returns:
        Configured SmartStudyAgent instance
    """
    return SmartStudyAgent(api_key=api_key)


def build_advanced_agent(api_key: str, model_name: str = "mistralai/mistral-7b-instruct") -> SmartStudyAgent:
    """
    Build an advanced Smart Study Agent with custom model.
    
    Args:
        api_key: OpenRouter API key
        model_name: Custom model name
    
    Returns:
        Configured SmartStudyAgent instance
    """
    return SmartStudyAgent(
        api_key=api_key,
        model_name=model_name,
        max_iterations=7,
        verbose=True
    )
