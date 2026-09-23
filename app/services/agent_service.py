from pydantic import ValidationError

from app.core.config import client
from app.tools.agent_tools import tools,tool_registry

def run_agent(question: str,max_steps: int = 5) -> str:
    messages = [
        {"role":"user","content":question}
    ]
    for step in range(max_steps):
        # print(f"======Step {step+1}:======")
        response = client.chat.completions.create(
            model="deepseek-flash",
            messages=messages,
            tools=tools
        )
        ai_message = response.choices[0].message
        # print("AI:",ai_message)
        messages.append(ai_message)
        if not ai_message.tool_calls:
            return ai_message.content
        for tool_call in ai_message.tool_calls:
            tool_name = tool_call.function.name
            tool_info = tool_registry.get(tool_name)
            if tool_info is None:
                result = f"Tool {tool_name} not found"
            else:
                tool_function = tool_info["function"]
                args_model = tool_info["args_model"]
                try:
                    validated_args = args_model.model_validate_json(tool_call.function.arguments)
                    result = tool_function(**validated_args.model_dump())
                except ValidationError as e:
                    result = f"Validation error: {e}"
                except Exception as e:
                    result = f"Tool execution failed:: {e}"
            # print("tool_message:",result)
            messages.append(
                {
                    "role":"tool",
                    "tool_call_id": tool_call.id,
                    "content":str(result)
                }
            )
    return "Agent 执行超过最大步数"