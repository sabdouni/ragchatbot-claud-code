# RAG Chatbot Query Flow Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend<br/>(script.js)
    participant API as FastAPI<br/>(app.py)
    participant RAG as RAGSystem<br/>(rag_system.py)
    participant SM as SessionManager
    participant AI as AIGenerator<br/>(ai_generator.py)
    participant TM as ToolManager
    participant ST as CourseSearchTool
    participant VS as VectorStore<br/>(ChromaDB)
    participant Claude as Claude API

    U->>F: Types query & clicks send
    F->>F: Disable input, show loading
    F->>API: POST /api/query<br/>{query, session_id}
    
    API->>RAG: rag_system.query(query, session_id)
    
    RAG->>SM: get_conversation_history(session_id)
    SM-->>RAG: conversation history
    
    RAG->>AI: generate_response(prompt, history, tools)
    
    AI->>Claude: Request with system prompt<br/>& tool definitions
    
    Note over Claude: Claude analyzes query<br/>Decides if search needed
    
    alt Course-specific question
        Claude->>AI: Tool call: search_course_content
        AI->>TM: execute_tool(search_course_content)
        TM->>ST: search(query)
        ST->>VS: search_content(query, max_results=5)
        VS-->>ST: Relevant chunks + metadata
        ST-->>TM: Search results + sources
        TM-->>AI: Tool response
        AI->>Claude: Tool results
        Claude-->>AI: Generated response using context
    else General question
        Claude-->>AI: Direct response (no search)
    end
    
    AI-->>RAG: Final response
    RAG->>TM: get_last_sources()
    TM-->>RAG: Sources list
    RAG->>TM: reset_sources()
    RAG->>SM: add_exchange(session_id, query, response)
    
    RAG-->>API: (response, sources)
    API-->>F: JSON: {answer, sources, session_id}
    
    F->>F: Remove loading, display response
    F->>U: Shows answer + sources
```

## Flow Breakdown

### **Phase 1: User Interaction**
1. User types query in chat input
2. Frontend disables input, shows loading animation
3. Makes POST request to `/api/query` endpoint

### **Phase 2: API Processing**
4. FastAPI receives request, validates data
5. Creates/retrieves session ID
6. Calls RAG system for processing

### **Phase 3: RAG Orchestration**
7. RAG system retrieves conversation history
8. Prepares prompt for AI generator
9. Passes tools (search capabilities) to AI

### **Phase 4: AI Processing**
10. AI generator sends request to Claude API
11. Claude analyzes query type using system prompt
12. **Decision Point**: Course-specific vs general question

### **Phase 5A: Vector Search (if needed)**
13. Claude calls search tool via tool manager
14. CourseSearchTool queries ChromaDB vector store
15. Returns semantically similar course chunks
16. Claude synthesizes response using retrieved context

### **Phase 5B: Direct Response (if general)**
13. Claude answers directly without search

### **Phase 6: Response Assembly**
14. AI generator returns final response
15. RAG system extracts sources from tool usage
16. Updates conversation history for session
17. Returns response and sources

### **Phase 7: Frontend Display**
18. Frontend receives JSON response
19. Updates UI with AI answer and sources
20. Re-enables input for next query

## Key Components

- **Session Management**: Maintains conversation context
- **Tool-Augmented AI**: Claude decides when to search
- **Vector Search**: Semantic matching for relevant content
- **Source Attribution**: Tracks material references
- **Real-time UX**: Loading states and smooth interactions