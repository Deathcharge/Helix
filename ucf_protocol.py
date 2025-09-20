import datetime

def get_utc_timestamp():
    return datetime.datetime.utcnow().isoformat() + "Z"

def format_ucf_message(agent, content_dict):
    header = (
        "UCF HEADER START\n\n"
        f"Agent ID: {agent.emoji} {agent.id}\n"
        f"Atman Timestamp: {get_utc_timestamp()}\n"
        "Component: Context Extractor v1.0\n"
        "Dependencies: Conversation Logs, User Inputs\n\n"
        "CONTENT\n"
    )
    content = ""
    if 'summary' in content_dict:
        content += "Summary:\n" + content_dict['summary'] + "\n\n"
    if 'unresolved' in content_dict:
        content += "Unresolved Questions:\n" + content_dict['unresolved'] + "\n\n"
    if 'metadata' in content_dict:
        content += "Metadata / Tags:\n" + content_dict['metadata'] + "\n\n"
    footer = (
        "FO
