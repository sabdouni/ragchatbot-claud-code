import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Tool Usage Guidelines:
- **Course Content Search**: Use `search_course_content` for questions about specific course topics, concepts, or detailed educational materials
- **Course Outline**: Use `get_course_outline` for questions about course structure, lesson lists, or course overviews
- **Sequential tool usage**: You can make up to 2 rounds of tool calls if needed for comprehensive answers
- Use tools strategically: search first, then outline/refine as needed for complex queries
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Course content questions**: Use search tool first, then answer
- **Course structure/outline questions**: Use outline tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the outline tool"

For course outline responses, always include:
1. Course title
2. Course link (if available)
3. Complete lesson list with lesson numbers and titles

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def _build_system_content(self, conversation_history: Optional[str] = None) -> str:
        """Build system content with optional conversation history."""
        return (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
    
    def _make_api_call(self, messages: List[Dict[str, Any]], system_content: str, 
                       tools: Optional[List] = None):
        """Make API call to Claude with error handling."""
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        return self.client.messages.create(**api_params)
    
    def _execute_tools_and_update_messages(self, response, messages: List[Dict[str, Any]], 
                                          tool_manager) -> List[Dict[str, Any]]:
        """Execute tools from response and update messages with results."""
        # Add AI's tool use response
        messages = messages.copy()
        messages.append({"role": "assistant", "content": response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, 
                        **content_block.input
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })
                except Exception as e:
                    # Add error as tool result to continue conversation
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Tool execution failed: {str(e)}"
                    })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        return messages
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to 2 rounds of sequential tool calling.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build initial messages and system content
        messages = [{"role": "user", "content": query}]
        system_content = self._build_system_content(conversation_history)
        
        # Sequential tool calling loop - maximum 2 rounds
        for round_num in range(1, 3):
            # Provide tools only on first round to prevent infinite loops
            current_tools = tools if round_num == 1 else None
            
            # Make API call
            response = self._make_api_call(messages, system_content, current_tools)
            
            # Check termination conditions
            if response.stop_reason != "tool_use" or not tool_manager:
                # No tool use - return response and terminate
                return response.content[0].text
            
            # Execute tools and update messages for next round
            messages = self._execute_tools_and_update_messages(response, messages, tool_manager)
            
            # If we're at max rounds, make final call without tools
            if round_num == 2:
                final_response = self._make_api_call(messages, system_content, None)
                return final_response.content[0].text
        
        # Safety fallback (should never reach here)
        return "Error: Maximum rounds exceeded"
    
