"""Thin boundary to the shared provider; no credential loading/cache duplication."""
async def structured_detection(**kwargs):
    from pathlib import Path
    from providers.gemini import make_adapter, get_cache, UsageLog
    root = Path(kwargs["run_dir"])
    adapter = make_adapter(run_dir=root, cache=get_cache(root / "cache"),
        usage=UsageLog(root / "logs" / "detect_usage.jsonl"),
        config={"temperature":0.0,"media_resolution":"high"})
    return await adapter(**kwargs)
