"""End-to-end integration tests for GoodMem LlamaIndex tool spec.

These tests require a running GoodMem API server and valid credentials.
Set environment variables GOODMEM_API_KEY, GOODMEM_BASE_URL, and
GOODMEM_TEST_PDF_PATH before running.

Usage:
    GOODMEM_API_KEY=gm_xxx GOODMEM_BASE_URL=https://localhost:8080 \
    GOODMEM_TEST_PDF_PATH=/path/to/test.pdf \
    python -m pytest tests/test_tools_goodmem.py -v -s
"""

import os
import time

import pytest

from llama_index.tools.goodmem import GoodMemToolSpec

API_KEY = os.environ.get("GOODMEM_API_KEY", "")
BASE_URL = os.environ.get("GOODMEM_BASE_URL", "")
PDF_PATH = os.environ.get("GOODMEM_TEST_PDF_PATH", "")


@pytest.fixture(scope="module")
def tool_spec():
    """Create a GoodMemToolSpec instance for testing."""
    if not API_KEY or not BASE_URL:
        pytest.skip("GOODMEM_API_KEY and GOODMEM_BASE_URL must be set")
    return GoodMemToolSpec(
        api_key=API_KEY,
        base_url=BASE_URL,
        verify_ssl=False,
    )


@pytest.fixture(scope="module")
def embedder_id(tool_spec):
    """Get the first available embedder ID."""
    embedders = tool_spec.list_embedders()
    assert len(embedders) > 0, "No embedders available"
    eid = embedders[0].get("embedderId") or embedders[0].get("id")
    assert eid, "Could not get embedder ID"
    return eid


@pytest.fixture(scope="module")
def space_id(tool_spec, embedder_id):
    """Create a test space and return its ID."""
    result = tool_spec.create_space(
        name=f"llama-index-test-{int(time.time())}",
        embedder_id=embedder_id,
    )
    assert result["success"] is True, f"Failed to create space: {result}"
    return result["spaceId"]


class TestGoodMemToolSpec:
    """End-to-end tests for GoodMemToolSpec."""

    def test_list_embedders(self, tool_spec):
        """Test listing available embedders."""
        embedders = tool_spec.list_embedders()
        assert isinstance(embedders, list)
        assert len(embedders) > 0
        first = embedders[0]
        assert "embedderId" in first or "id" in first

    def test_list_spaces(self, tool_spec):
        """Test listing available spaces."""
        spaces = tool_spec.list_spaces()
        assert isinstance(spaces, list)

    def test_create_space(self, tool_spec, embedder_id):
        """Test creating a new space."""
        result = tool_spec.create_space(
            name=f"llama-index-create-test-{int(time.time())}",
            embedder_id=embedder_id,
        )
        assert result["success"] is True
        assert "spaceId" in result
        assert result["reused"] is False

    def test_create_space_reuse(self, tool_spec, embedder_id):
        """Test that creating a space with the same name reuses existing."""
        name = f"llama-index-reuse-test-{int(time.time())}"
        result1 = tool_spec.create_space(name=name, embedder_id=embedder_id)
        assert result1["success"] is True

        result2 = tool_spec.create_space(name=name, embedder_id=embedder_id)
        assert result2["success"] is True
        assert result2["reused"] is True
        assert result2["spaceId"] == result1["spaceId"]

    def test_create_memory_text(self, tool_spec, space_id):
        """Test creating a memory with plain text content."""
        result = tool_spec.create_memory(
            space_id=space_id,
            text_content="The quick brown fox jumps over the lazy dog. "
            "This is a test memory for the LlamaIndex GoodMem integration.",
        )
        assert result["success"] is True
        assert "memoryId" in result
        assert result["contentType"] == "text/plain"

    def test_create_memory_pdf(self, tool_spec, space_id):
        """Test creating a memory with a PDF file."""
        if not PDF_PATH:
            pytest.skip("GOODMEM_TEST_PDF_PATH not set")
        result = tool_spec.create_memory(
            space_id=space_id,
            file_path=PDF_PATH,
        )
        assert result["success"] is True, f"Failed to create PDF memory: {result}"
        assert "memoryId" in result
        assert result["contentType"] == "application/pdf"

    def test_get_memory(self, tool_spec, space_id):
        """Test fetching a specific memory by ID."""
        # Create a memory first
        create_result = tool_spec.create_memory(
            space_id=space_id,
            text_content="Memory to be fetched for testing get_memory.",
        )
        assert create_result["success"] is True
        memory_id = create_result["memoryId"]

        result = tool_spec.get_memory(memory_id=memory_id)
        assert result["success"] is True
        assert "memory" in result

    def test_retrieve_memories(self, tool_spec, space_id):
        """Test retrieving memories via semantic search."""
        # Create a memory with specific content
        tool_spec.create_memory(
            space_id=space_id,
            text_content="Artificial intelligence is transforming healthcare "
            "by enabling early disease detection through medical imaging analysis.",
        )

        result = tool_spec.retrieve_memories(
            query="AI in healthcare",
            space_ids=[space_id],
            max_results=5,
            wait_for_indexing=True,
        )
        assert result["success"] is True
        assert "results" in result
        assert result["totalResults"] > 0

    def test_delete_memory(self, tool_spec, space_id):
        """Test deleting a memory."""
        create_result = tool_spec.create_memory(
            space_id=space_id,
            text_content="Memory to be deleted for testing.",
        )
        assert create_result["success"] is True
        memory_id = create_result["memoryId"]

        result = tool_spec.delete_memory(memory_id=memory_id)
        assert result["success"] is True
        assert result["memoryId"] == memory_id

    def test_to_tool_list(self, tool_spec):
        """Test that tool spec converts to a list of FunctionTools."""
        tools = tool_spec.to_tool_list()
        assert len(tools) == 7
        tool_names = {t.metadata.name for t in tools}
        expected = {
            "create_space",
            "create_memory",
            "retrieve_memories",
            "get_memory",
            "delete_memory",
            "list_spaces",
            "list_embedders",
        }
        assert tool_names == expected
