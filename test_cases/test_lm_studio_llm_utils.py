import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalystOU.utils.llm_utils import (
    build_completion_payload,
    create_async_openai_client,
    get_llm_config,
)


async def main() -> None:
    model_name = "qwen/qwen3.5-35b-a3b"
    config = get_llm_config(model_name)
    print("CONFIG", json.dumps(config, indent=2))

    client = create_async_openai_client(config)
    payload = build_completion_payload(
        model_name=model_name,
        system_prompt="You are a helpful assistant.",
        user_prompt="Reply with exactly: OK",
        config=config,
    )
    print("PAYLOAD", json.dumps(payload, indent=2))

    response = await client.chat.completions.create(**payload)
    text = getattr(getattr(response.choices[0], "message", None), "content", None)
    print("CHOICES", len(response.choices))
    print("TEXT", text)


if __name__ == "__main__":
    asyncio.run(main())
