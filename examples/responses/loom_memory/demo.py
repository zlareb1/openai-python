"""Opt-in live example: this makes billable model calls and real Loom requests."""

from __future__ import annotations

import os
import asyncio
import argparse

from loom import Agent as LoomAgent

from openai import AsyncOpenAI
from examples.responses.loom_memory.memory import PreferenceMemory, respond


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("prompt")
    args = parser.parse_args()
    async with (
        AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]) as client,
        LoomAgent(
            org=os.environ["LOOM_ORG"],
            namespace=os.environ["LOOM_NAMESPACE"],
            agent="responses-preferences",
            base_url=os.environ["LOOM_BASE_URL"],
            api_key=os.environ["LOOM_API_KEY"],
            mentor_brief=False,
            auto_run=False,
            coalesce_sets=False,
        ) as agent,
    ):
        response = await respond(
            client,
            PreferenceMemory(agent, user_id=os.environ["LOOM_USER_ID"]),
            model=os.environ["OPENAI_MODEL"],
            prompt=args.prompt,
        )
        print(response.output_text)


if __name__ == "__main__":
    asyncio.run(main())
