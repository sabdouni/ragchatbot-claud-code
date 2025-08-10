from unittest.mock import MagicMock, Mock, patch

import pytest
from config import Config
from models import Course
from rag_system import RAGSystem


class TestRAGIntegration:
    """Integration tests for the RAG system"""

    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_rag_system_initialization(
        self, mock_doc_proc, mock_vector_store, mock_ai_gen, mock_session_mgr
    ):
        """Test RAG system component initialization"""
        config = Config()
        rag_system = RAGSystem(config)

        # Verify all components are initialized
        assert rag_system.document_processor is not None
        assert rag_system.vector_store is not None
        assert rag_system.ai_generator is not None
        assert rag_system.session_manager is not None
        assert rag_system.tool_manager is not None

        # Verify tools are registered
        assert len(rag_system.tool_manager.tools) == 2  # search + outline tools
        assert "search_course_content" in rag_system.tool_manager.tools
        assert "get_course_outline" in rag_system.tool_manager.tools

    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_query_without_session(
        self, mock_doc_proc, mock_vector_store, mock_ai_gen, mock_session_mgr
    ):
        """Test query processing without session ID"""
        config = Config()
        rag_system = RAGSystem(config)

        # Mock AI generator response
        mock_ai_gen.return_value.generate_response.return_value = "Test AI response"

        # Mock tool manager to have no sources
        rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        rag_system.tool_manager.reset_sources = Mock()

        response, sources = rag_system.query("What is Python?")

        assert response == "Test AI response"
        assert sources == []

        # Verify AI generator was called
        mock_ai_gen.return_value.generate_response.assert_called_once()
        call_args = mock_ai_gen.return_value.generate_response.call_args[1]
        assert "What is Python?" in call_args["query"]
        assert call_args["tools"] is not None
        assert call_args["tool_manager"] is not None

    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_query_with_session(
        self, mock_doc_proc, mock_vector_store, mock_ai_gen, mock_session_mgr
    ):
        """Test query processing with session ID"""
        config = Config()
        rag_system = RAGSystem(config)

        # Mock session manager
        mock_session_mgr.return_value.get_conversation_history.return_value = (
            "Previous conversation"
        )
        mock_session_mgr.return_value.add_exchange = Mock()

        # Mock AI generator response
        mock_ai_gen.return_value.generate_response.return_value = "Session response"

        # Mock tool manager
        rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        rag_system.tool_manager.reset_sources = Mock()

        response, sources = rag_system.query("Follow-up question", "session_123")

        assert response == "Session response"

        # Verify session operations
        mock_session_mgr.return_value.get_conversation_history.assert_called_once_with(
            "session_123"
        )
        mock_session_mgr.return_value.add_exchange.assert_called_once_with(
            "session_123", "Follow-up question", "Session response"
        )

        # Verify AI generator got conversation history
        call_args = mock_ai_gen.return_value.generate_response.call_args[1]
        assert call_args["conversation_history"] == "Previous conversation"

    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_query_with_sources(
        self, mock_doc_proc, mock_vector_store, mock_ai_gen, mock_session_mgr
    ):
        """Test query processing that returns sources from tools"""
        config = Config()
        rag_system = RAGSystem(config)

        # Mock AI generator response
        mock_ai_gen.return_value.generate_response.return_value = (
            "Response with sources"
        )

        # Mock tool manager with sources
        mock_sources = [
            {"text": "Course A - Lesson 1", "link": "https://example.com/lesson1"},
            {"text": "Course B - Lesson 2", "link": None},
        ]
        rag_system.tool_manager.get_last_sources = Mock(return_value=mock_sources)
        rag_system.tool_manager.reset_sources = Mock()

        response, sources = rag_system.query("Search query")

        assert response == "Response with sources"
        assert sources == mock_sources

        # Verify sources were retrieved and reset
        rag_system.tool_manager.get_last_sources.assert_called_once()
        rag_system.tool_manager.reset_sources.assert_called_once()

    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_add_course_document_success(
        self,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_session_mgr,
        sample_course,
        sample_course_chunks,
    ):
        """Test successful course document addition"""
        config = Config()
        rag_system = RAGSystem(config)

        # Mock document processor
        mock_doc_proc.return_value.process_course_document.return_value = (
            sample_course,
            sample_course_chunks,
        )

        # Mock vector store operations
        mock_vector_store.return_value.add_course_metadata = Mock()
        mock_vector_store.return_value.add_course_content = Mock()

        course, chunk_count = rag_system.add_course_document("/path/to/course.txt")

        assert course == sample_course
        assert chunk_count == len(sample_course_chunks)

        # Verify document processing
        mock_doc_proc.return_value.process_course_document.assert_called_once_with(
            "/path/to/course.txt"
        )

        # Verify vector store operations
        mock_vector_store.return_value.add_course_metadata.assert_called_once_with(
            sample_course
        )
        mock_vector_store.return_value.add_course_content.assert_called_once_with(
            sample_course_chunks
        )

    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_add_course_document_error(
        self, mock_doc_proc, mock_vector_store, mock_ai_gen, mock_session_mgr
    ):
        """Test course document addition with error"""
        config = Config()
        rag_system = RAGSystem(config)

        # Mock document processor to raise error
        mock_doc_proc.return_value.process_course_document.side_effect = Exception(
            "Parse error"
        )

        course, chunk_count = rag_system.add_course_document("/invalid/path.txt")

        assert course is None
        assert chunk_count == 0

    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    @patch('os.path.exists')
    @patch('os.path.isfile')
    @patch('os.listdir')
    def test_add_course_folder_success(
        self,
        mock_listdir,
        mock_isfile,
        mock_exists,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_session_mgr,
        sample_course,
        sample_course_chunks,
    ):
        """Test successful course folder addition"""
        config = Config()
        rag_system = RAGSystem(config)

        # Mock file system
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.txt", "course2.pdf", "ignored.jpg"]
        mock_isfile.return_value = True  # All files are considered files

        # Mock existing course titles
        mock_vector_store.return_value.get_existing_course_titles.return_value = []

        # Mock document processing with different courses for each file
        course1 = sample_course
        course2 = Course(
            title="Different Course",
            course_link="https://example.com/course2",
            instructor="Test Instructor 2",
            lessons=sample_course.lessons,
        )
        mock_doc_proc.return_value.process_course_document.side_effect = [
            (course1, sample_course_chunks),
            (course2, sample_course_chunks),
        ]

        # Mock vector store operations
        mock_vector_store.return_value.add_course_metadata = Mock()
        mock_vector_store.return_value.add_course_content = Mock()

        total_courses, total_chunks = rag_system.add_course_folder("/docs")

        assert total_courses == 2  # txt and pdf files
        assert total_chunks == len(sample_course_chunks) * 2

        # Verify processing was called for valid files only
        assert mock_doc_proc.return_value.process_course_document.call_count == 2

    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_get_course_analytics(
        self, mock_doc_proc, mock_vector_store, mock_ai_gen, mock_session_mgr
    ):
        """Test course analytics retrieval"""
        config = Config()
        rag_system = RAGSystem(config)

        # Mock vector store analytics
        mock_vector_store.return_value.get_course_count.return_value = 5
        mock_vector_store.return_value.get_existing_course_titles.return_value = [
            "Course A",
            "Course B",
            "Course C",
            "Course D",
            "Course E",
        ]

        analytics = rag_system.get_course_analytics()

        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Course A" in analytics["course_titles"]

    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_error_handling_in_query(
        self, mock_doc_proc, mock_vector_store, mock_ai_gen, mock_session_mgr
    ):
        """Test error handling during query processing"""
        config = Config()
        rag_system = RAGSystem(config)

        # Mock AI generator to raise error
        mock_ai_gen.return_value.generate_response.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            rag_system.query("Test query")

    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_tool_integration(
        self, mock_doc_proc, mock_vector_store, mock_ai_gen, mock_session_mgr
    ):
        """Test that tools are properly integrated with AI generator"""
        config = Config()
        rag_system = RAGSystem(config)

        # Mock AI generator response
        mock_ai_gen.return_value.generate_response.return_value = "Tool-based response"

        # Mock tool manager
        rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        rag_system.tool_manager.reset_sources = Mock()

        response, sources = rag_system.query("What is machine learning?")

        # Verify AI generator was called with tools
        call_args = mock_ai_gen.return_value.generate_response.call_args[1]
        assert call_args["tools"] is not None
        assert len(call_args["tools"]) == 2  # search + outline tools
        assert call_args["tool_manager"] is rag_system.tool_manager

        # Verify tool definitions are correct
        tools = call_args["tools"]
        tool_names = [tool["name"] for tool in tools]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names
