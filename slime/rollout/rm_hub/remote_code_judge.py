import os
import re
import random
import asyncio
import aiohttp
import json

import os, json, re, aiohttp

async def partial_credit_judge(response: str, label: str) -> float:
    judgeHost = os.getenv("JUDGE_HOST", "LightCPVerifier")
    judgePort = os.getenv("JUDGE_PORT", "8081")
    judgeBaseUrl = f"http://{judgeHost}:{judgePort}"

    pid = label[:5]
    if len(label) > 5 and label[5] not in [' ', '_']:
        pid = label[:6]

    code, lang = _extract_code(response)
    if code:
        sid = await _submit_code(pid=pid, lang=lang, code=code, baseUrl=judgeBaseUrl)
        if sid:
            res = await _wait_for_result(sid, judgeBaseUrl)
            if res:
                return 1.0
    else:
        return -0.1  # No code found, return a negative score to indicate failure

    try:
        gpt_api = os.getenv("GPT_API", "https://api.openai.com/v1/responses")
        gpt_key = os.getenv("GPT_KEY", "")

        prompt = f"""
You are a competitive programming expert, and are very good at evaluating the quality of solutions. 
Here a newbee gives a solution to a problem but it failed to pass all test cases.
Your job is to evaluate how good the solution is, and give a score between 0.0 to 1.0 (inclusive).
Here is his code solution:
{code}
And here for your reference, is the problem statement and the standard solution:
{label}
Please analyze the solution by identifying the key ideas, match them with the reference, 
and give a float score that reflects how far the submitted solution deviates from the standard one.
Your final answer should be ONLY the float number, nothing else.
"""

        async with aiohttp.ClientSession() as session:
            headers = {"Content-Type": "application/json"}
            if gpt_key:
                headers["Authorization"] = f"Bearer {gpt_key}"            

            payload = {
                "model": "o4-mini",
                "reasoning": {"effort": "high"},
                "input": prompt
            }

            async with session.post(gpt_api, headers=headers, data=json.dumps(payload)) as r:
                if r.status == 200:
                    data = await r.json()
                    # Responses API 输出在 data["output"] 中
                    text_output = ""
                    for item in data.get("output", []):
                        if item.get("type") == "message":
                            parts = item.get("content", [])
                            for p in parts:
                                if p.get("type") == "output_text":
                                    text_output += p.get("text", "")
                    raw_output = text_output.strip()

                    try:
                        score = float(re.findall(r"[-+]?\d*\.?\d+", raw_output)[0])
                        return min(max(score, 0.0), 1.0) * 0.8
                    except Exception:
                        return 0.0
                else:
                    return 0.0
    except Exception:
        return 0.0

    return 0.0

async def remote_code_judge(response: str, label: str) -> float:
    """提取 C++ 代码 -> 提交到评测机 -> 等待结果 -> 返回部分分数"""
    judgeHost = os.getenv("JUDGE_HOST", "LightCPVerifier")
    judgePort = os.getenv("JUDGE_PORT", "8081")
    judgeBaseUrl = f"http://{judgeHost}:{judgePort}"

    code, lang = _extract_code(response)
    if not code:
        return -1.0

    sid = await _submit_code(pid = label, lang = lang, code = code, baseUrl = judgeBaseUrl)
    if not sid:
        return 0.0

    try:
        async with aiohttp.ClientSession() as session:
            for _ in range(30):
                async with session.get(f"{judgeBaseUrl}/result/{sid}") as r:
                    if r.status == 200:
                        data = await r.json()
                        if data.get("status") == "done":
                            if data.get("passed", False):
                                return 1.0
                            if data.get("cases"):
                                total_cases = len(data["cases"])
                                passed_cases = total_cases - 1
                                return 0.015 * passed_cases

                await asyncio.sleep(1)
    except Exception as e:
        print(f"Query failed: {e}")
    return 0.0
        
async def remote_01_code_judge(response: str, label: str) -> float:
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
    pattern_with_lang = r"```([^\s`\n]+)[ \t]*\r?\n?([\s\S]*?)\r?\n?```"
    matches = re.finditer(pattern_with_lang, response, re.DOTALL)
    for match in matches:
        lang = match.group(1).lower()
        code = match.group(2).strip()
        code_blocks.append((code, lang))
    
    # 匹配无语言标识的代码块
    pattern_no_lang = r"```(?![A-Za-z])[\t ]*\r?\n?([\s\S]*?)\r?\n?```"
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