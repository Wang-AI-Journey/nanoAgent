import os
import json
import subprocess
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL")
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Execute a bash command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
]


def execute_bash(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr


def read_file(path):
    with open(path, "r") as f:
        return f.read()


def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)
    return f"Wrote to {path}"


functions = {"execute_bash": execute_bash, "read_file": read_file, "write_file": write_file}


def chat(messages, user_message, max_iterations=5):
    # 把新消息追加到历史，而不是重置
    messages.append({"role": "user", "content": user_message})

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            tools=tools,
        )
        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"[Tool] {name}({args})")
            result = functions[name](**args) if name in functions else f"Error: Unknown tool '{name}'"
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

    return "Max iterations reached"


if __name__ == "__main__":
    # messages 在整个会话中保持，上下文跨轮次保留
    messages = [{"role": "system", "content": "You are a helpful assistant. Be concise."}]

    print("多轮对话模式，输入 exit 退出，输入 clear 清空上下文\n")
    while True:
        user_input = input("你: ").strip()
        if user_input.lower() == "exit":
            break
        if not user_input:
            continue
        reply = chat(messages, user_input)
        print(f"AI: {reply}\n")
