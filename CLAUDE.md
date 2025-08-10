# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Important: This project uses `uv` for ALL dependency management and Python command execution. Never use `pip`, `pip install`, `python -m`, or direct Python commands. Always use `uv` equivalents.**

### Running the Application
```bash
# Quick start (recommended)
chmod +x run.sh
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Environment Setup
```bash
# Install dependencies
uv sync

# Create .env file with:
ANTHROPIC_API_KEY=your_api_key_here
```

### Package Management
```bash
# Sync/install all dependencies (equivalent to pip install -r requirements.txt)
uv sync

# Add new dependency (equivalent to pip install package_name)
uv add package_name

# Remove dependency (equivalent to pip uninstall package_name)  
uv remove package_name

# Run Python scripts (instead of python script.py)
uv run python script.py

# Run any Python module (instead of python -m module)
uv run python -m module

# Install specific versions
uv add "package_name==1.0.0"

# Install development dependencies
uv add --dev package_name

# Show installed packages (instead of pip list)
uv pip list

# Export requirements (instead of pip freeze)
uv pip freeze
```

### Code Quality & Development Tools

**Quick Commands:**
```bash
# Run complete quality pipeline (format, lint, test)
./scripts/quality.sh

# Check code quality without making changes
./scripts/check.sh

# Format code only
./scripts/format.sh

# Run linting only  
./scripts/lint.sh

# Run tests only
./scripts/test.sh
```

**Individual Tool Commands:**
```bash
# Code formatting with black
uv run black backend/

# Import sorting with isort  
uv run isort backend/

# Style checking with flake8
uv run flake8 backend/

# Type checking with mypy
uv run mypy backend/ --ignore-missing-imports

# Run tests with pytest
cd backend && uv run pytest -v
```

**Before Committing:**
- Always run `./scripts/quality.sh` to ensure code meets standards
- Use `./scripts/check.sh` to verify without making changes
- All quality checks must pass before committing code

### Application Access
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) chatbot system** for course materials with a tool-augmented AI approach.

### Core Architecture Pattern
The system uses **tool-augmented RAG** where Claude AI decides when to search the knowledge base:
1. User queries are sent to Claude with search tool definitions
2. Claude analyzes query type and decides whether to search course materials
3. If search is needed, Claude calls `CourseSearchTool` to query ChromaDB vector store
4. Claude synthesizes retrieved chunks into contextual responses

### Key Components

**RAG System (`backend/rag_system.py`)**
- Main orchestrator coordinating all components
- Entry point: `query(query, session_id)` method
- Manages tool-based AI interaction and session state

**Document Processing Pipeline**
- `DocumentProcessor`: Parses course files with structured format (title, instructor, lessons)
- Intelligent sentence-based chunking with overlap (800 chars, 100 char overlap)
- Adds contextual prefixes: `"Course [title] Lesson [number] content: [chunk]"`

**Vector Storage (`backend/vector_store.py`)**
- ChromaDB for persistent vector storage
- Uses SentenceTransformer embeddings (`all-MiniLM-L6-v2`)
- Stores `CourseChunk` objects with course/lesson metadata

**AI Generator (`backend/ai_generator.py`)**
- Interfaces with Claude API (`claude-sonnet-4-20250514`)
- System prompt instructs Claude on tool usage patterns
- Tool manager handles search tool execution

**Session Management**
- Maintains conversation context with configurable history (2 messages default)
- Session-based continuity across queries

### Data Models (`backend/models.py`)
- `Course`: Title, instructor, lessons, course link
- `Lesson`: Number, title, lesson link  
- `CourseChunk`: Content with course/lesson metadata for vector storage

### Expected Document Format
Course files should follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: Introduction
[content...]

Lesson 1: Topic Name  
Lesson Link: [optional_link]
[content...]
```

### Configuration (`backend/config.py`)
Key settings that affect behavior:
- `CHUNK_SIZE: 800` - Text chunk size for vector storage
- `CHUNK_OVERLAP: 100` - Character overlap between chunks
- `MAX_RESULTS: 5` - Vector search result limit
- `MAX_HISTORY: 2` - Conversation context length
- `CHROMA_PATH: "./chroma_db"` - Vector database location

### Startup Behavior
The application automatically loads documents from `docs/` folder on startup (`app.py:88-99`). Existing courses are skipped to avoid duplication.

### Frontend Integration
- Simple HTML/CSS/JS frontend served as static files
- Real-time chat interface with loading states
- Displays AI responses with expandable source citations
- Session persistence across page reloads