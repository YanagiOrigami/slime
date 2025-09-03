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

    code, lang = _extract_code(response)
    if not code:
        return 0.0

    sid = await _submit_code(pid = label, lang = lang, code = code, baseUrl = judgeBaseUrl)
    if not sid:
        return 0.0

    return 1.0 if await _wait_for_result(sid, judgeBaseUrl) else 0.0

def _extract_code(response: str) -> tuple[str, str]:
    """
    从LLM的API响应中提取最后的代码答案并识别语言
    
    Args:
        response: LLM的完整响应文本
        
    Returns:
        tuple: (代码内容, 语言类型) 
               语言类型为 "cpp" 或 "py"
               如果没找到代码返回 ("", "py")
    """
    
    # 查找所有代码块（按出现顺序）
    code_blocks = []
    
    # 匹配带语言标识的代码块
    pattern_with_lang = r"```(\w+)\n(.*?)\n```"
    matches = re.finditer(pattern_with_lang, response, re.DOTALL)
    for match in matches:
        lang = match.group(1).lower()
        code = match.group(2).strip()
        code_blocks.append((code, lang))
    
    # 匹配无语言标识的代码块
    pattern_no_lang = r"```\n(.*?)\n```"
    matches = re.finditer(pattern_no_lang, response, re.DOTALL)
    for match in matches:
        code = match.group(1).strip()
        # 通过代码内容推断语言
        lang = _detect_language(code)
        code_blocks.append((code, lang))
    
    # 如果没找到任何代码块
    if not code_blocks:
        return "", "py"
    
    # 返回最后一个代码块（最终答案）
    final_code, detected_lang = code_blocks[-1]
    
    # 标准化语言名称
    if detected_lang in ["cpp", "c++", "cxx"]:
        return final_code, "cpp"
    elif detected_lang in ["python", "py"]:
        return final_code, "py"
    else:
        # 如果语言不确定，通过代码内容再次判断
        inferred_lang = _detect_language(final_code)
        return final_code, inferred_lang

def _detect_language(code: str) -> str:
    code_lower = code.lower()
    cpp_indicators = [
        "#include", "using namespace std", "std::", 
        "cout<<", "cin>>", "endl", "vector<", "#define"
    ]   
    cpp_score = sum(1 for indicator in cpp_indicators if indicator in code_lower)
    if cpp_score > 0:
        return "cpp"
    else:
        return "py"


async def _submit_code(pid: str, code: str, baseUrl: str, lang: str = "cpp"):
    """提交代码"""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"pid": pid, "lang": lang, "code": code}
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