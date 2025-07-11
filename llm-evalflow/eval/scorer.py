# eval/scorer.py

def evaluate_response(response: str) -> dict:
    score = 0
    feedback = []

    if len(response.split()) > 5:
        score += 1
    else:
        feedback.append("Too short")

    if "I don't know" in response:
        feedback.append("Uncertain answer")
    else:
        score += 1

    return {"score": score, "feedback": feedback}
