
import os
import base64
import requests

from dotenv import load_dotenv
from .memory import Hindsight

load_dotenv()


class IncidentResponseAgent:

    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "qwen/qwen3.8-27b"
        self.memory = Hindsight()

        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is missing")

    def handle_incident(
        self,
        incident_description,
        image_inputs=None,
        log_texts=None
    ):
        print("\n[Agent] Analyzing incident:", incident_description)

        image_inputs = image_inputs or []
        log_texts = log_texts or []

        try:
            # 1. Retrieve historical incident context
            past_memories = []

            try:
                recalled = self.memory.recall(
                    query=incident_description,
                    limit=3
                )

                if hasattr(recalled, "results"):
                    past_memories = [
                        item.text for item in recalled.results
                    ]

            except Exception as error:
                print("[Agent] Memory recall warning:", error)

            # 2. Combine incident description and log evidence
            evidence_text = "\n\n".join(
                f"Log evidence {index + 1}:\n{log}"
                for index, log in enumerate(log_texts)
            )

            prompt = f"""
You are RunBook AI, an expert incident response assistant.

INCIDENT DESCRIPTION:
{incident_description}

UPLOADED LOG EVIDENCE:
{evidence_text if evidence_text else "No log files uploaded."}

RELEVANT HISTORICAL INCIDENTS:
{past_memories if past_memories else "No relevant past incidents found."}

IMAGE EVIDENCE:
Analyze all attached screenshots. Extract visible errors,
timestamps, metrics, service names and warning messages.

TASK:
1. Identify the likely root cause.
2. Separate confirmed evidence from hypotheses.
3. Use relevant historical incidents where applicable.
4. Provide immediate mitigation steps.
5. Provide permanent corrective actions.
6. Provide verification and monitoring steps.
7. Never invent configuration values or claim an action
   has been performed.
8. Require human approval before production changes.

Clearly explain which recommendations are supported
by historical incidents.

Generate a structured, practical remediation plan.
"""

            # 3. Construct multimodal message
            content = [
                {
                    "type": "text",
                    "text": prompt
                }
            ]

            for mime_type, image_bytes in image_inputs:
                encoded_image = base64.b64encode(
                    image_bytes
                ).decode("utf-8")

                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            f"data:{mime_type};base64,"
                            f"{encoded_image}"
                        )
                    }
                })

            # 4. Call Groq
            message_content = (
                content if image_inputs else prompt
            )

            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": message_content
                    }
                ],
                "temperature": 0.3
            }

            response = requests.post(
                self.groq_url,
                json=payload,
                headers=headers,
                timeout=90
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Groq API error {response.status_code}: "
                    f"{response.text[:1000]}"
                )

            result = response.json()

            remediation_plan = (
                result["choices"][0]["message"]["content"]
            )

            if not remediation_plan:
                raise RuntimeError(
                    "Groq returned an empty remediation plan."
                )

            # 5. Retain incident context in Hindsight
            self.memory.retain(
                text=(
                    f"Incident: {incident_description}\n"
                    f"Log evidence: {evidence_text}\n"
                    f"Remediation plan: {remediation_plan}"
                ),
                metadata={
                    "record_type": "incident",
                    "plan": remediation_plan
                }
            )

            print("\n[Agent] Remediation plan generated.")
            return remediation_plan

        finally:
            self.memory.close()


if __name__ == "__main__":
    agent = IncidentResponseAgent()

    try:
        agent.handle_incident(
            "High CPU utilization spike detected on production server."
        )
    finally:
        pass
