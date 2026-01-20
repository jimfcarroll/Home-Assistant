from google.adk.agents.llm_agent import ToolUnion

from config import DEBUG

def before_model_log_callback(agent: str, **kwargs) -> None:
    """
    Runs right before the model is called.

    You can:
    - inspect / modify the model_input
    - block by returning an escalation/action (see ADK docs/tutorial patterns)
    """
    if DEBUG:
        try:
            print(f"\n======== before_model_callback ({agent}) ==========")
            print(kwargs)
            print("========================================", flush=True)
        except Exception:
            pass
    else:
        print(f"before_model_log_callback Agent: {agent}", flush=True)

def before_tool_log_callback(agent: str, tool : ToolUnion, args : dict, tool_context, **kwargs) -> None:
    """
    Runs right before a tool is invoked.

    You can:
    - inspect / modify tool args
    - block tool calls by returning an error-shaped tool result (tutorial pattern)
    """
    if DEBUG:
        try:
            print(f"\n======== before_tool_callback ({agent})==========")
            print(kwargs)
            print("=======================================", flush=True)
        except Exception:
            pass
    else:
        print(f"before_tool_log_callback Agent: {agent}", flush=True)

