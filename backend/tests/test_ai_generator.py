import pytest
from unittest.mock import Mock, patch, MagicMock
import anthropic
from ai_generator import AIGenerator


class TestAIGenerator:
    """Test cases for AIGenerator"""
    
    def test_init(self):
        """Test AIGenerator initialization"""
        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        
        assert generator.model == "claude-sonnet-4-20250514"
        assert generator.base_params["model"] == "claude-sonnet-4-20250514"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800
    
    @patch('anthropic.Anthropic')
    def test_generate_response_without_tools(self, mock_anthropic_class):
        """Test basic response generation without tools"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response"
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client
        
        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        result = generator.generate_response("What is Python?")
        
        assert result == "Test response"
        mock_client.messages.create.assert_called_once()
        
        # Verify API call parameters
        call_args = mock_client.messages.create.call_args[1]
        assert call_args["model"] == "claude-sonnet-4-20250514"
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["content"] == "What is Python?"
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic_class):
        """Test response generation with conversation history"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Follow-up response"
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client
        
        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        result = generator.generate_response(
            "What about JavaScript?", 
            conversation_history="User: What is Python?\nAssistant: Python is a programming language."
        )
        
        assert result == "Follow-up response"
        
        # Verify system prompt includes history
        call_args = mock_client.messages.create.call_args[1]
        assert "Previous conversation:" in call_args["system"]
        assert "Python is a programming language" in call_args["system"]
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic_class):
        """Test response with tools available but no tool use"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Direct answer without tools"
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client
        
        mock_tool_manager = Mock()
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        
        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        result = generator.generate_response(
            "What is 2+2?", 
            tools=tools, 
            tool_manager=mock_tool_manager
        )
        
        assert result == "Direct answer without tools"
        
        # Verify tools were provided in API call
        call_args = mock_client.messages.create.call_args[1]
        assert "tools" in call_args
        assert call_args["tool_choice"] == {"type": "auto"}
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_tool_execution(self, mock_anthropic_class):
        """Test response generation with tool execution"""
        mock_client = Mock()
        
        # First response: tool use
        mock_tool_response = Mock()
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.input = {"query": "Python basics"}
        tool_block.id = "tool_123"
        mock_tool_response.content = [tool_block]
        mock_tool_response.stop_reason = "tool_use"
        
        # Second response: final answer
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Based on search: Python is great!"
        mock_final_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        mock_anthropic_class.return_value = mock_client
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Python is a programming language used for..."
        
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        
        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        result = generator.generate_response(
            "Tell me about Python", 
            tools=tools, 
            tool_manager=mock_tool_manager
        )
        
        assert result == "Based on search: Python is great!"
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", 
            query="Python basics"
        )
        
        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2
    
    @patch('anthropic.Anthropic')
    def test_handle_tool_execution_multiple_tools(self, mock_anthropic_class):
        """Test handling multiple tool calls in one response"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Create initial response with multiple tool uses
        tool_block1 = Mock()
        tool_block1.type = "tool_use"
        tool_block1.name = "search_course_content"
        tool_block1.input = {"query": "Python"}
        tool_block1.id = "tool_1"
        
        tool_block2 = Mock()
        tool_block2.type = "tool_use"
        tool_block2.name = "get_course_outline"
        tool_block2.input = {"course_title": "Python Course"}
        tool_block2.id = "tool_2"
        
        initial_response = Mock()
        initial_response.content = [tool_block1, tool_block2]
        
        # Mock final response
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Combined response"
        
        mock_client.messages.create.return_value = final_response
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Python content result",
            "Python course outline"
        ]
        
        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        result = generator._handle_tool_execution(
            initial_response, 
            {"messages": [{"role": "user", "content": "test"}], "system": "test"},
            mock_tool_manager
        )
        
        assert result == "Combined response"
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="Python")
        mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_title="Python Course")
    
    @patch('anthropic.Anthropic')
    def test_api_error_handling(self, mock_anthropic_class):
        """Test handling of API errors"""
        mock_client = Mock()
        # Create a simple exception instead of anthropic.APIError which has complex constructor
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic_class.return_value = mock_client
        
        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        
        with pytest.raises(Exception):
            generator.generate_response("Test query")
    
    def test_system_prompt_content(self):
        """Test system prompt includes proper tool usage instructions"""
        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        
        assert "Course Content Search" in generator.SYSTEM_PROMPT
        assert "search_course_content" in generator.SYSTEM_PROMPT
        assert "Course Outline" in generator.SYSTEM_PROMPT
        assert "get_course_outline" in generator.SYSTEM_PROMPT
        assert "One tool call per query maximum" in generator.SYSTEM_PROMPT
    
    @patch('anthropic.Anthropic')
    def test_tool_execution_error_handling(self, mock_anthropic_class):
        """Test handling of tool execution errors"""
        mock_client = Mock()
        
        # Tool use response
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.input = {"query": "test"}
        tool_block.id = "tool_123"
        
        mock_tool_response = Mock()
        mock_tool_response.content = [tool_block]
        mock_tool_response.stop_reason = "tool_use"
        
        # Final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Error response"
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        mock_anthropic_class.return_value = mock_client
        
        # Tool manager that returns error
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution failed: Database error"
        
        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        result = generator.generate_response(
            "Search query", 
            tools=[{"name": "search_course_content"}], 
            tool_manager=mock_tool_manager
        )
        
        assert result == "Error response"
        
        # Verify the tool result was passed to the second API call
        second_call_args = mock_client.messages.create.call_args_list[1][1]
        messages = second_call_args["messages"]
        
        # Should have user message, assistant tool use, and user tool results
        assert len(messages) == 3
        assert messages[2]["role"] == "user"
        assert "Tool execution failed: Database error" in str(messages[2]["content"])