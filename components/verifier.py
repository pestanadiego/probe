MAX_SCORE_THRESHOLD = 0.7
MEAN_SCORE_THRESHOLD = 0.5

def verify(scores):
    if not scores:
        return False
    max_score = max(scores)
    mean_score = sum(scores) / len(scores)
    return max_score >= MAX_SCORE_THRESHOLD and mean_score >= MEAN_SCORE_THRESHOLD
