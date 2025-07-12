from dotenv import load_dotenv
load_dotenv()
import os
import resend
from typing import List
from resend import Email

resend.api_key = os.environ.get("RESEND_API_KEY", "")

def send_success_email(to: list[str], agent_name: str, summary: str = "") -> Email:
    subject = f"✅ Workflow Completed: {agent_name}"
    html = f"""
        <h2>🎉 Your agent <code>{agent_name}</code> has successfully completed its workflow.</h2>
        <p>Everything went smoothly.</p>
        {"<p><strong>Summary:</strong><br>" + summary + "</p>" if summary else ""}
        <p>Visit the dashboard to review the results or start a new run.</p>
    """

    params: resend.Emails.SendParams = {
        "from": "Job Agent <varun@devhub.co.in>",
        "to": to,
        "subject": subject,
        "html": html,
    }
    return resend.Emails.send(params)

def send_error_email(to: list[str], agent_name: str, error_message: str = "") -> Email:
    subject = f"❌ Workflow Failed: {agent_name}"
    html = f"""
        <h2>🚨 Your agent <code>{agent_name}</code> encountered an error.</h2>
        <p>Unfortunately, the workflow could not complete as expected.</p>
        {"<p><strong>Error Details:</strong><br>" + error_message + "</p>" if error_message else ""}
        <p>Please check the agent configuration or logs and try again.</p>
    """

    params: resend.Emails.SendParams = {
        "from": "Job Agent <varun@devhub.co.in>",
        "to": to,
        "subject": subject,
        "html": html,
    }
    return resend.Emails.send(params)

