"""
This contains a simple script to calculate the matching score after inference
"""

def evaluate(answer: str, predicted: str):
	"""
	Evaluate if the predicted answer contains the true answer
	:param answer: true answer (gold label)
	:param predicted: generated answer (predicted label)
	:return: True/False (boolean)
	"""

	print(predicted)

	return predicted in list(answer)


def accuracy(evaluated):
	"""
	Calculate the accuracy
	:param evaluated:
	"""
	return evaluated.count(True) / len(evaluated) 