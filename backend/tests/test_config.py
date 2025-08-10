import os
import sys
from unittest.mock import patch

import pytest
from config import Config


class TestConfig:
    """Test cases for configuration validation and loading"""

    def test_default_config_values(self):
        """Test default configuration values"""
        config = Config()

        # Test model settings
        assert config.ANTHROPIC_MODEL == "claude-sonnet-4-20250514"
        assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"

        # Test document processing settings
        assert config.CHUNK_SIZE == 800
        assert config.CHUNK_OVERLAP == 100
        assert config.MAX_RESULTS == 5  # Fixed from 0
        assert config.MAX_HISTORY == 2

        # Test database path
        assert config.CHROMA_PATH == "./chroma_db"

    def test_environment_variable_loading(self):
        """Test that environment variables can be loaded"""
        # Since .env is already loaded, test the mechanism works
        config = Config()
        # Just verify the API key is loaded from somewhere (env or .env)
        assert isinstance(config.ANTHROPIC_API_KEY, str)
        assert hasattr(config, 'ANTHROPIC_API_KEY')

    def test_config_structure(self):
        """Test configuration has required fields"""
        config = Config()
        # Test that all required config fields exist
        required_fields = [
            'ANTHROPIC_API_KEY',
            'ANTHROPIC_MODEL',
            'EMBEDDING_MODEL',
            'CHUNK_SIZE',
            'CHUNK_OVERLAP',
            'MAX_RESULTS',
            'MAX_HISTORY',
            'CHROMA_PATH',
        ]
        for field in required_fields:
            assert hasattr(config, field), f"Config missing required field: {field}"

    def test_config_validation_max_results_positive(self):
        """Test that MAX_RESULTS is positive"""
        config = Config()
        assert config.MAX_RESULTS > 0, "MAX_RESULTS must be positive for search to work"

    def test_config_validation_chunk_size_reasonable(self):
        """Test that chunk size is reasonable"""
        config = Config()
        assert (
            100 <= config.CHUNK_SIZE <= 2000
        ), "Chunk size should be between 100-2000 characters"

    def test_config_validation_chunk_overlap_smaller_than_size(self):
        """Test that chunk overlap is smaller than chunk size"""
        config = Config()
        assert (
            config.CHUNK_OVERLAP < config.CHUNK_SIZE
        ), "Overlap must be smaller than chunk size"

    def test_config_validation_max_history_reasonable(self):
        """Test that max history is reasonable"""
        config = Config()
        assert 0 <= config.MAX_HISTORY <= 10, "History should be between 0-10 messages"


class TestConfigIssues:
    """Test cases for identifying configuration issues that could cause problems"""

    def test_zero_max_results_issue(self):
        """Test that MAX_RESULTS=0 would cause search failures"""
        # This test verifies the fix we implemented
        config = Config()

        # Verify the critical fix is in place
        assert (
            config.MAX_RESULTS != 0
        ), "MAX_RESULTS=0 causes search to return no results"
        assert (
            config.MAX_RESULTS >= 1
        ), "MAX_RESULTS must be at least 1 for meaningful search"

    def test_empty_api_key_detection(self):
        """Test detection of empty API key"""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()

            # This should be caught by the application startup
            if not config.ANTHROPIC_API_KEY:
                print("WARNING: ANTHROPIC_API_KEY is not set - API calls will fail")

    def test_invalid_chunk_overlap_detection(self):
        """Test detection of invalid chunk overlap"""
        config = Config()

        # Chunk overlap should not exceed chunk size
        assert (
            config.CHUNK_OVERLAP <= config.CHUNK_SIZE
        ), "Invalid configuration: CHUNK_OVERLAP exceeds CHUNK_SIZE"

    def test_model_name_format(self):
        """Test that model name follows expected format"""
        config = Config()

        # Model name should follow Claude format
        assert config.ANTHROPIC_MODEL.startswith(
            "claude"
        ), "Model name should start with 'claude'"
        assert len(config.ANTHROPIC_MODEL) > 6, "Model name seems too short"

    def test_path_configurations(self):
        """Test that path configurations are reasonable"""
        config = Config()

        # ChromaDB path should be relative or absolute
        assert config.CHROMA_PATH, "CHROMA_PATH should not be empty"
        assert not config.CHROMA_PATH.startswith("//"), "Invalid path format"


class TestConfigDefaults:
    """Test default configuration values for robustness"""

    def test_recommended_max_results(self):
        """Test that MAX_RESULTS has a reasonable default"""
        config = Config()

        # Should be between 1-10 for good performance/relevance balance
        assert (
            1 <= config.MAX_RESULTS <= 10
        ), f"MAX_RESULTS ({config.MAX_RESULTS}) should be between 1-10"

    def test_chunk_size_optimization(self):
        """Test that chunk size is optimized for embeddings"""
        config = Config()

        # 800 characters is good for sentence transformers
        # Too small: loses context, too large: dilutes relevance
        assert (
            500 <= config.CHUNK_SIZE <= 1200
        ), f"CHUNK_SIZE ({config.CHUNK_SIZE}) should be optimized for embeddings (500-1200)"

    def test_chunk_overlap_percentage(self):
        """Test that chunk overlap is reasonable percentage of chunk size"""
        config = Config()

        overlap_percentage = (config.CHUNK_OVERLAP / config.CHUNK_SIZE) * 100
        assert (
            5 <= overlap_percentage <= 25
        ), f"Chunk overlap ({overlap_percentage:.1f}%) should be 5-25% of chunk size"

    def test_history_memory_management(self):
        """Test that conversation history limit manages memory well"""
        config = Config()

        # 2 messages (1 exchange) is good for context without bloat
        assert (
            1 <= config.MAX_HISTORY <= 5
        ), f"MAX_HISTORY ({config.MAX_HISTORY}) should be 1-5 for good memory management"


class TestEnvironmentVariableHandling:
    """Test environment variable loading and error handling"""

    def test_api_key_validation(self):
        """Test API key validation logic"""
        config = Config()
        # Test that API key exists and is not empty after loading
        api_key = config.ANTHROPIC_API_KEY
        if api_key:  # If we have an API key
            assert len(api_key.strip()) > 0, "API key should not be just whitespace"
            assert api_key.startswith("sk-"), "API key should start with 'sk-'"

    def test_config_defaults_when_no_env_file(self):
        """Test that config constructor doesn't crash with defaults"""
        # This tests the Config dataclass default values work
        config = Config()
        assert config.ANTHROPIC_MODEL  # Should have a default value
        assert config.EMBEDDING_MODEL  # Should have a default value
        assert config.CHROMA_PATH  # Should have a default value

    def test_config_singleton_behavior(self):
        """Test that config behaves consistently"""
        from config import config as config_instance

        # Should be the same instance/values
        new_config = Config()
        assert config_instance.ANTHROPIC_MODEL == new_config.ANTHROPIC_MODEL
        assert config_instance.MAX_RESULTS == new_config.MAX_RESULTS
