import json


def openhands_messages(traj):
    out = []
    for m in traj:
        c = m.get("content") or ""
        if m["role"] == "assistant":
            for tc in m.get("tool_calls") or []:
                f = tc["function"]
                try:
                    a = json.loads(f["arguments"])
                except (ValueError, TypeError):
                    a = f["arguments"]
                c += "\n<tool_call>\n" + json.dumps({"name": f["name"], "arguments": a}) + "\n</tool_call>"
            out.append({"role": "assistant", "content": c})
        elif m["role"] == "tool":
            out.append({"role": "user", "content": f"<tool_response>\n{c}\n</tool_response>"})
        else:
            out.append({"role": m["role"], "content": c})
    return out
