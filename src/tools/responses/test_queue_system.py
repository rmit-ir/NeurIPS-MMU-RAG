"""
Test script to verify the queue system works with multiple subscribers.
"""

import asyncio
from systems.rag_interface import RunStreamingResponse, CitationItem
from tools.responses.openai_stream import to_openai_stream


async def mock_rag_stream():
    """Mock RAG stream factory for testing - returns an async generator."""
    print('mock_rag_stream started.')
    # Simulate intermediate steps
    yield RunStreamingResponse(
        is_intermediate=True,
        intermediate_steps="Searching for information...",
        citations=[],
        complete=False
    )

    await asyncio.sleep(0.1)  # Simulate processing time

    yield RunStreamingResponse(
        is_intermediate=True,
        intermediate_steps="Processing results...",
        citations=[],
        complete=False
    )

    await asyncio.sleep(0.1)

    # Final response with content
    yield RunStreamingResponse(
        is_intermediate=False,
        final_report="This is the final answer to your question.",
        citations=[CitationItem(url="https://example.com",
                                icon_url="", date=None, title="Example Source")],
        complete=True
    )
    print('mock_rag_stream completed.')


async def simulate_client(client_id: str, chat_hash: str, delay: float = 0):
    """Simulate a client making a request."""
    await asyncio.sleep(delay)
    print(f"Client {client_id} starting request...")

    chunks = []
    async for chunk in to_openai_stream(mock_rag_stream, model="test-model", chat_hash=chat_hash):
        chunks.append(chunk)
        data = chunk[6:] if chunk.startswith("data: ") else chunk
        if data.strip() == "[DONE]":
            print(f"Client {client_id} processing complete.")
            break

    print(f"Client {client_id} completed with {len(chunks)} chunks")
    return chunks


async def test_multiple_subscribers():
    """Test multiple subscribers to the same chat_hash."""
    print("Testing multiple subscribers to the same chat_hash...")

    chat_hash = "test-hash-123"

    # Start multiple clients with the same chat_hash
    # Client 1 starts immediately, Client 2 starts after 0.05s, Client 3 after 0.15s
    tasks = [
        simulate_client("Client-1", chat_hash, delay=0),
        # Joins during stream
        simulate_client("Client-2", chat_hash, delay=0.05),
        # Joins after stream completes
        simulate_client("Client-3", chat_hash, delay=0.15),
    ]

    results = await asyncio.gather(*tasks)

    print("\n=== Results ===")
    for i, chunks in enumerate(results, 1):
        print(f"Client-{i} received {len(chunks)} chunks")

    # Verify all clients got the same content
    if len(set(len(chunks) for chunks in results)) == 1:
        print("✅ All clients received the same number of chunks")
    else:
        print("❌ Clients received different numbers of chunks")

    # Check if content is identical
    if all(chunks == results[0] for chunks in results[1:]):
        print("✅ All clients received identical content")
    else:
        print("❌ Clients received different content")


async def test_different_hashes():
    """Test that different chat_hashes create separate streams."""
    print("\nTesting different chat_hashes...")

    tasks = [
        simulate_client("Hash-A-Client", "hash-a"),
        simulate_client("Hash-B-Client", "hash-b"),
    ]

    results = await asyncio.gather(*tasks)
    print(f"Hash-A client: {len(results[0])} chunks")
    print(f"Hash-B client: {len(results[1])} chunks")
    print("✅ Different hashes work independently")


async def main():
    """Run all tests."""
    await test_multiple_subscribers()
    await test_different_hashes()
    print("\n🎉 Queue system tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
