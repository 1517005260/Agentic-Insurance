"""Runners: long-running per-request work that streams events back via SSE.

A runner is the bridge between (a) the synchronous, pre-existing algorithm
layer (``RAGPipeline``, ``BaseAgent``, ``ProofAgent``) and (b) the async
FastAPI route that needs to stream incremental progress to the browser.

The pattern is::

    bus = EventBus()
    asyncio.get_running_loop().run_in_executor(
        None, lambda: pipeline.run(query, on_event=bus.push)
    )
    async for chunk in bus.stream():
        yield chunk
"""
