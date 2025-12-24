"""English text refining agent using LangChain"""

import os
from typing import Optional
from langchain.agents import create_agent


class TextRefinerAgent:
    """Agent specialized in refining transcribed English text"""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        api_key: Optional[str] = None
    ):
        """
        Initialize the text refining agent using LangChain's create_agent.
        
        Args:
            model: OpenAI model to use (default: gpt-4o-mini)
            temperature: Temperature for text generation (default: 0.3 for more focused output)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Store API key for later use if needed
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        
        # System prompt for the agent
        system_prompt = """You are an expert English text refinement agent specialized in processing transcribed speech.

Your task is to refine transcribed text that may contain:
- Chunk overlaps (repeated phrases from audio processing)
- Broken or incomplete sentences
- Transcription errors
- Disfluencies (ums, ahs, false starts)
- Missing punctuation or capitalization
- Control commands (e.g., "stop alfred", "hey alfred" or similar stop/start phrases)

Instructions:
1. Understand the full context and intent of the text
2. Remove duplicate content from overlapping chunks
3. Remove any control commands or trigger phrases (such as "stop alfred", "hey alfred", or similar variations)
4. Correct any obvious transcription errors
5. Fix grammar, punctuation, and capitalization
6. Preserve the original meaning and tone
7. Make the text natural and readable
8. Return ONLY the refined text - no explanations, no markdown, no additional commentary

Important: Strip out any stop commands, start commands, or similar control phrases that may have been transcribed as part of the instruction text.

Output format: Return the refined text directly, nothing else."""
        
        # Create agent using LangChain's create_agent
        # Model format: "openai:model-name" for OpenAI models
        # No tools needed for text refinement - it's a straightforward task
        model_string = f"openai:{model}"
        self.agent = create_agent(
            model=model_string,
            tools=[],  # No tools needed for text refinement
            system_prompt=system_prompt
        )
    
    def _extract_refined_text(self, response) -> str:
        """
        Extract refined text from agent response.
        
        Args:
            response: The response from the agent
        
        Returns:
            Extracted refined text
        """
        # The response structure may vary, so we handle it safely
        if isinstance(response, dict):
            messages = response.get("messages", [])
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, "content"):
                    refined_text = last_message.content.strip()
                elif isinstance(last_message, dict):
                    refined_text = last_message.get("content", "").strip()
                else:
                    refined_text = str(last_message).strip()
            else:
                refined_text = str(response).strip()
        else:
            refined_text = str(response).strip()
        
        return refined_text
    
    def refine(self, instruction: str) -> str:
        """
        Refine the transcribed instruction text.
        
        Args:
            instruction: The raw transcribed text to refine
        
        Returns:
            Refined and corrected text
        """
        if not instruction or not instruction.strip():
            return ""
        
        # Invoke the agent with the instruction
        response = self.agent.invoke({
            "messages": [{"role": "user", "content": instruction.strip()}]
        })
        
        return self._extract_refined_text(response)
    
    async def refine_async(self, instruction: str) -> str:
        """
        Async version of refine method.
        
        Args:
            instruction: The raw transcribed text to refine
        
        Returns:
            Refined and corrected text
        """
        if not instruction or not instruction.strip():
            return ""
        
        # Invoke the agent asynchronously
        response = await self.agent.ainvoke({
            "messages": [{"role": "user", "content": instruction.strip()}]
        })
        
        return self._extract_refined_text(response)

