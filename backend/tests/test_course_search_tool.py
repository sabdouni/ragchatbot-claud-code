import pytest
from unittest.mock import Mock, patch
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test cases for CourseSearchTool"""
    
    def test_get_tool_definition(self, mock_vector_store):
        """Test tool definition structure"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]
    
    def test_execute_basic_query(self, mock_vector_store):
        """Test basic query execution"""
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        # Should call vector store search
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=None
        )
        
        # Should return formatted results
        assert "Test Course" in result
        assert "Test content from course" in result
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Test Course - Lesson 1"
    
    def test_execute_with_course_filter(self, mock_vector_store):
        """Test query with course name filter"""
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", course_name="Test Course")
        
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="Test Course",
            lesson_number=None
        )
        assert "Test Course" in result
    
    def test_execute_with_lesson_filter(self, mock_vector_store):
        """Test query with lesson number filter"""
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", lesson_number=1)
        
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=1
        )
        assert "Test Course" in result
    
    def test_execute_empty_results(self, mock_vector_store, empty_search_results):
        """Test handling of empty search results"""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("no results query")
        
        assert result == "No relevant content found."
        assert len(tool.last_sources) == 0
    
    def test_execute_empty_results_with_filters(self, mock_vector_store, empty_search_results):
        """Test empty results with filter information"""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("no results query", course_name="Test Course", lesson_number=1)
        
        assert "No relevant content found in course 'Test Course' in lesson 1." in result
    
    def test_execute_error_handling(self, mock_vector_store, error_search_results):
        """Test error handling from vector store"""
        mock_vector_store.search.return_value = error_search_results
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("error query")
        
        assert "Search error: Database connection failed" in result
        assert len(tool.last_sources) == 0
    
    def test_format_results_with_links(self, mock_vector_store):
        """Test result formatting with lesson links"""
        # Create search results with lesson links
        results_with_links = SearchResults(
            documents=["Content with link"],
            metadata=[{
                "course_title": "Test Course",
                "lesson_number": 2,
                "lesson_link": "https://example.com/lesson2"
            }],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = results_with_links
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        assert "[Test Course - Lesson 2]" in result
        assert "Content with link" in result
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["link"] == "https://example.com/lesson2"
    
    def test_format_results_without_lesson_number(self, mock_vector_store):
        """Test result formatting without lesson numbers"""
        results_no_lesson = SearchResults(
            documents=["General course content"],
            metadata=[{
                "course_title": "Test Course",
                "lesson_number": None,
                "lesson_link": None
            }],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = results_no_lesson
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        assert "[Test Course]" in result
        assert "General course content" in result
        assert tool.last_sources[0]["text"] == "Test Course"
        assert tool.last_sources[0]["link"] is None
    
    def test_multiple_results_formatting(self, mock_vector_store):
        """Test formatting of multiple search results"""
        multiple_results = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course 1", "lesson_number": 1, "lesson_link": None},
                {"course_title": "Course 2", "lesson_number": 2, "lesson_link": "https://example.com/lesson2"}
            ],
            distances=[0.1, 0.2]
        )
        mock_vector_store.search.return_value = multiple_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        assert "[Course 1 - Lesson 1]" in result
        assert "[Course 2 - Lesson 2]" in result
        assert "Content 1" in result
        assert "Content 2" in result
        assert len(tool.last_sources) == 2
        

class TestToolManager:
    """Test cases for ToolManager"""
    
    def test_register_tool(self, mock_vector_store):
        """Test tool registration"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        
        manager.register_tool(tool)
        
        assert "search_course_content" in manager.tools
        assert len(manager.get_tool_definitions()) == 1
    
    def test_execute_tool(self, mock_vector_store):
        """Test tool execution through manager"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        result = manager.execute_tool("search_course_content", query="test query")
        
        assert "Test Course" in result
        mock_vector_store.search.assert_called_once()
    
    def test_execute_nonexistent_tool(self):
        """Test executing tool that doesn't exist"""
        manager = ToolManager()
        result = manager.execute_tool("nonexistent_tool", query="test")
        
        assert "Tool 'nonexistent_tool' not found" in result
    
    def test_get_last_sources(self, mock_vector_store):
        """Test getting sources from last search"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        # Execute search to generate sources
        manager.execute_tool("search_course_content", query="test query")
        sources = manager.get_last_sources()
        
        assert len(sources) == 1
        assert sources[0]["text"] == "Test Course - Lesson 1"
    
    def test_reset_sources(self, mock_vector_store):
        """Test resetting sources"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        # Execute search and verify sources exist
        manager.execute_tool("search_course_content", query="test query")
        assert len(manager.get_last_sources()) == 1
        
        # Reset and verify sources are cleared
        manager.reset_sources()
        assert len(manager.get_last_sources()) == 0