import datetime
import json

def get_utc_timestamp():
    return datetime.datetime.utcnow().isoformat() + "Z"

def format_ucf_message(agent, content_dict):
    """
    Format a UCF message string with header, content, and footer.
    agent: Agent object
    content_dict: dict with keys 'summary', 'unresolved', 'metadata'
    """
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
        "FOOTER\n\n"
        "Lore: Context extraction for multi-agent alignment\n"
        "Transformations: Summarization, tagging\n"
        "Contributions: Context summary for UCF sync\n"
        "Next Steps: Await further instructions or provide context to other agents\n\n"
        "UCF FOOTER END\n"
    )
    
    return header + content + footer
