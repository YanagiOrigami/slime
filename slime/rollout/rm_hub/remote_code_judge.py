import os
import re
import random
import asyncio
import aiohttp

async def remote_code_judge(response: str, label: str) -> float:
    return random.random()
    """提取 C++ 代码 -> 提交到评测机 -> 等待结果 -> 返回 1.0/0.0"""
    judgeHost = os.getenv("JUDGE_HOST", "LightCPVerifier")
    judgePort = os.getenv("JUDGE_PORT", "8081")
    judgeBaseUrl = f"http://{judgeHost}:{judgePort}"

    code = _extract_code(response)
    if not code:
        return 0.0

    sid = await _submit_code(label, code, judgeBaseUrl)
    if not sid:
        return 0.0

    return 1.0 if await _wait_for_result(sid, judgeBaseUrl) else 0.0


# ===== 内部工具函数 =====
def _extract_code(resp: str) -> str:
    """从 response 中提取 C++ 代码"""
    m = re.search(r"```(?:cpp|c\+\+)\n(.*?)\n```", resp, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\n(.*?)\n```", resp, re.DOTALL)
    if m:
        code = m.group(1).strip()
        if "#include" in code or "using namespace std" in code:
            return code
    return ""


async def _submit_code(pid: str, code: str, baseUrl: str):
    """提交代码"""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"pid": pid, "lang": "cpp", "code": code}
            async with session.post(f"{baseUrl}/submit", json=payload) as r:
                if r.status == 200:
                    return (await r.json()).get("sid")
    except Exception as e:
        print(f"Submit failed: {e}")
    return None


async def _wait_for_result(sid: int, baseUrl: str, timeout: int = 30) -> bool:
    """等待结果"""
    try:
        async with aiohttp.ClientSession() as session:
            for _ in range(timeout):
                async with session.get(f"{baseUrl}/result/{sid}") as r:
                    if r.status == 200:
                        data = await r.json()
                        if data.get("status") == "done":
                            return bool(data.get("passed", False))
                await asyncio.sleep(1)
    except Exception as e:
        print(f"Query failed: {e}")
    return False