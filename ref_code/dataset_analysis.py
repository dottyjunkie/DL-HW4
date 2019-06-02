import numpy as np
import matplotlib.pyplot as plt 

with open('en.txt', 'r', encoding='utf-8') as f:
	en_lines = f.read().split('\n')
	# print('num_lines:{}'.format(len(en_lines)))

with open('fr.txt', 'r', encoding='utf-8-sig') as f:
	fr_lines = f.read().split('\n')




def plot_hist(lines):
	n_words = []
	for line in lines:
		n_words.append(len(line))
	plt.clf()
	plt.hist(n_words, bins=100)
	plt.show()

def foo():
	en_min = 60
	en_max = 80
	fr_min = 60
	fr_max = 90

	trimmed_en_lines = []
	trimmed_fr_lines = []
	for i, (en_line, fr_line) in enumerate(zip(en_lines, fr_lines)):
		if len(en_line)>en_min and len(en_line)<en_max and len(fr_line)>fr_min and len(fr_line)<fr_max:
			trimmed_en_lines.append(en_line)
			trimmed_fr_lines.append(fr_line)
	print('Saved percentage: {}%'.format(len(trimmed_en_lines) / len(en_lines)))
	print('Saved percentage: {}%'.format(len(trimmed_fr_lines) / len(fr_lines)))

en_words = []
for line in en_lines:
	en_words.append(len(line.split()))
plt.clf()
plt.hist(en_words, bins=20)
plt.show()