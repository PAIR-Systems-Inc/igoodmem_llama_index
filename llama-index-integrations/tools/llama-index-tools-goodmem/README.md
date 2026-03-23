# LlamaIndex Tools Integration: GoodMem

GoodMem is a memory layer for AI agents with support for semantic storage, retrieval, and summarization. This package exposes GoodMem operations as LlamaIndex tools that can be used with any LlamaIndex agent.

## Installation

```bash
pip install llama-index-tools-goodmem
```

## Configuration

You need a GoodMem API key and base URL to use this integration.

```python
from llama_index.tools.goodmem import GoodMemToolSpec

tool_spec = GoodMemToolSpec(
    api_key="your-goodmem-api-key",
    base_url="https://api.goodmem.ai",
)
```

### Parameters

| Parameter    | Type   | Required | Description                                      |
|-------------|--------|----------|--------------------------------------------------|
| `api_key`   | str    | Yes      | Your GoodMem API key (X-API-Key)                 |
| `base_url`  | str    | Yes      | Base URL of your GoodMem API server              |
| `verify_ssl`| bool   | No       | Whether to verify SSL certificates (default: True)|

## Available Tools

This integration exposes the following tools:

### create_space

Create a new space or reuse an existing one. A space is a logical container for organizing related memories, configured with embedders that convert text to vector embeddings.

```python
result = tool_spec.create_space(
    name="my-space",
    embedder_id="embedder-uuid",
    chunk_size=256,
    chunk_overlap=25,
    keep_strategy="KEEP_END",
    length_measurement="CHARACTER_COUNT",
)
```

### create_memory

Store a document as a new memory in a space. Accepts a file path or plain text.

```python
# Text content
result = tool_spec.create_memory(
    space_id="space-uuid",
    text_content="Your text content here.",
)

# File (PDF, DOCX, image, etc.)
result = tool_spec.create_memory(
    space_id="space-uuid",
    file_path="/path/to/document.pdf",
)
```

### retrieve_memories

Perform similarity-based semantic retrieval across one or more spaces. Returns matching chunks ranked by relevance.

```python
result = tool_spec.retrieve_memories(
    query="What is the main topic?",
    space_ids=["space-uuid-1", "space-uuid-2"],
    max_results=5,
    wait_for_indexing=True,
)
```

### get_memory

Fetch a specific memory record by its ID, including metadata, processing status, and optionally the original content.

```python
result = tool_spec.get_memory(
    memory_id="memory-uuid",
    include_content=True,
)
```

### delete_memory

Permanently delete a memory and its associated chunks and vector embeddings.

```python
result = tool_spec.delete_memory(memory_id="memory-uuid")
```

### list_spaces

List all available spaces in GoodMem.

```python
spaces = tool_spec.list_spaces()
```

### list_embedders

List all available embedder models in GoodMem.

```python
embedders = tool_spec.list_embedders()
```

## Usage with a LlamaIndex Agent

```python
from llama_index.tools.goodmem import GoodMemToolSpec
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

# Initialize tools
tool_spec = GoodMemToolSpec(
    api_key="your-api-key",
    base_url="https://api.goodmem.ai",
)
tools = tool_spec.to_tool_list()

# Create agent with GoodMem tools
llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

# Use the agent
response = agent.chat("Create a space called 'research' and store this text: ...")
```

## Testing

Run the integration tests with:

```bash
GOODMEM_API_KEY=your_key \
GOODMEM_BASE_URL=https://localhost:8080 \
GOODMEM_TEST_PDF_PATH=/path/to/test.pdf \
python -m pytest tests/test_tools_goodmem.py -v -s
```
