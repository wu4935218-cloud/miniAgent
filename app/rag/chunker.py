def split_text(
        text: str,
        chunk_size: int = 200,
        chunk_overlap: int = 40,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks

def split_by_paragraph(text: str) -> list[str]:
    return [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

def recursive_split_text(
        text: str,
        chunk_size: int = 200,
        separators: list[str] | None = None
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if separators is None:
        separators = [
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "：",
            "，",
            " "
        ]
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        return [
            text[i:i+chunk_size]
            for i in range(0,len(text),chunk_size)
        ]
    separator = separators[0]
    parts = text.split(separator)
    if len(parts) == 1:
        return recursive_split_text(
            text,chunk_size,separators[1:]
        )
    chunks = []
    current = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        candidate = current + separator + part if current else part
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.extend(
                recursive_split_text(
                    current,chunk_size,separators[1:]
                )
            )
        current = part
    if current:
        chunks.extend(
            recursive_split_text(
                current,chunk_size,separators[1:]
            )
        )
    return chunks

if __name__ == "__main__":
    text = """
Tool Calling 是大模型调用外部工具的一种机制。模型本身不会直接执行 Python 函数，而是生成工具调用请求，由程序负责执行。

FastAPI 是一个 Python Web 框架，可以用于把 Agent 封装成 HTTP 服务。

asyncio 是 Python 的异步编程框架，非常适合网络请求、数据库访问等 I/O 密集型场景。
"""

    chunks = recursive_split_text(
        text,
        chunk_size=80
    )

    for index, chunk in enumerate(chunks):
        print(
            f"\n====== Chunk {index} ======"
        )
        print(chunk)
        print(
            "length:",
            len(chunk)
        )
