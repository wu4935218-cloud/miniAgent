import asyncio
import time


async def test1():
    print("test1开始")
    await asyncio.sleep(5)
    print("test1结束")
    return 20
async def test2():
    print("test2开始")
    await asyncio.sleep(3)
    print("test2结束")
    return 10
async def main():
    print("main开始")
    # # 获取事件循环
    # event_loop = asyncio.get_event_loop()
    # # 手动创建任务
    # t1 = event_loop.create_task(test1())
    # t2 = event_loop.create_task(test2())
    #
    # result = await t1
    # print(result)
    # result = await t2
    # print(result)
    result = await asyncio.gather(test1(), test2())
    print(result)
    print("main结束")

start_time = time.time()
asyncio.run(main())
# # 创建事件循环
# event_loop = asyncio.get_event_loop()
# # 启动事件循环
# event_loop.run_until_complete(main())
print("总耗时：", time.time() - start_time)

