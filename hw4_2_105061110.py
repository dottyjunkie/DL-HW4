import numpy as np

with open('en.txt', 'r', encoding='utf-8') as f:
	lines = f.readlines() # 137760 lines

with open('en.txt', 'r', encoding='utf-8') as f:
	text = f.read()

words = set(text.split())
word2int = {word:i for i, word in enumerate(words)}
int2word = dict(enumerate(words))

line1 = lines[1].split()
line1_int = np.array([word2int[word] for word in line1], dtype=np.int32)
for word, code in zip(line1, line1_int):
	print("{}->{}".format(word, code))


