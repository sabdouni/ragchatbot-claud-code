import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add backend directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Course, CourseChunk, Lesson
from vector_store import SearchResults


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore for testing"""
    mock_store = Mock()
    mock_store.search.return_value = SearchResults(
        documents=["Test content from course"],
        metadata=[
            {
                "course_title": "Test Course",
                "lesson_number": 1,
                "lesson_link": "https://example.com/lesson1",
            }
        ],
        distances=[0.1],
    )
    mock_store._resolve_course_name.return_value = "Test Course"
    mock_store.get_all_courses_metadata.return_value = [
        {
            "title": "Test Course",
            "course_link": "https://example.com/course",
            "instructor": "Test Instructor",
            "lessons": [
                {
                    "lesson_number": 1,
                    "lesson_title": "Introduction",
                    "lesson_link": "https://example.com/lesson1",
                },
                {
                    "lesson_number": 2,
                    "lesson_title": "Advanced Topics",
                    "lesson_link": "https://example.com/lesson2",
                },
            ],
        }
    ]
    return mock_store


@pytest.fixture
def sample_course():
    """Sample course for testing"""
    lessons = [
        Lesson(
            lesson_number=1,
            title="Introduction",
            lesson_link="https://example.com/lesson1",
        ),
        Lesson(
            lesson_number=2,
            title="Advanced Topics",
            lesson_link="https://example.com/lesson2",
        ),
    ]
    return Course(
        title="Test Course",
        course_link="https://example.com/course",
        instructor="Test Instructor",
        lessons=lessons,
    )


@pytest.fixture
def sample_course_chunks():
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="This is test content from lesson 1",
            course_title="Test Course",
            lesson_number=1,
            lesson_link="https://example.com/lesson1",
            chunk_index=0,
        ),
        CourseChunk(
            content="This is test content from lesson 2",
            course_title="Test Course",
            lesson_number=2,
            lesson_link="https://example.com/lesson2",
            chunk_index=1,
        ),
    ]


@pytest.fixture
def empty_search_results():
    """Empty search results for testing"""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search_results():
    """Error search results for testing"""
    return SearchResults.empty("Search error: Database connection failed")


@pytest.fixture
def mock_anthropic_response():
    """Mock Anthropic API response"""
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = "This is a test response"
    mock_response.content[0].type = "text"
    mock_response.stop_reason = "end_turn"
    return mock_response


@pytest.fixture
def mock_anthropic_tool_response():
    """Mock Anthropic API response with tool use"""
    mock_response = Mock()

    # First content block is tool use
    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.name = "search_course_content"
    tool_block.input = {"query": "test query"}
    tool_block.id = "tool_123"

    mock_response.content = [tool_block]
    mock_response.stop_reason = "tool_use"
    return mock_response


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client"""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = "Test response"
    mock_response.stop_reason = "end_turn"
    mock_client.messages.create.return_value = mock_response
    return mock_client
