import numpy as np

def getScore(vals):
	c = np.array([np.log(x) for x in vals])
	print(c)
	return c
	
scores = np.array([
	[-1.2, -2.7, -3.2, -5.0, -5.8]
])

print(np.log(-1.2))

for r in scores:
	getScore(r)