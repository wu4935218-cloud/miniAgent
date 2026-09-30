import asyncio
import time

async def fake_tool(name: str):
    print(f"{name} start")
    await asyncio.sleep(2)
    print(f"{name} end")
    return name

async def main():
    start = time.perf_counter()

    result = await asyncio.gather(
        fake_tool("tool1"),
        fake_tool("tool2"),
        fake_tool("tool3")
    )

    end = time.perf_counter()
    print(result)
    print("time:",end - start)

asyncio.run(main())