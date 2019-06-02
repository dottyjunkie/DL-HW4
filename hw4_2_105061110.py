import numpy as np

# Good ref:
# utf-8-sig: https://stackoverflow.com/questions/17912307/u-ufeff-in-python-string


def print_word2int(lines, word2int):
	for line in lines:
		line = '<SOL> ' + line[:-1] + ' <EOL>'
		line = line.split()
		# print(line)
		line2int = np.array([word2int[word] for word in line], dtype=np.int32)
		word_len = []
		for i, word in enumerate(line):
			word_len.append(len(word))
			diff = len(str(line2int[i])) - len(word)
			if diff <= 0:
				n_space = 1
			else:
				n_space = diff + 1
			print(word + ' '*n_space, end='')
		print()
		for i, code in enumerate(line2int):
			print('{:<{width}}'.format(code, width=word_len[i]), end=' ')
		print()

def word_encoding(lan, n_lines=0):
	if lan == 'fr':
		encoding = 'utf-8-sig'
	else:
		encoding = 'utf-8'

	with open('{}.txt'.format(lan), 'r', encoding=encoding) as f:
		lines = f.readlines() # 137760 lines

	with open('{}.txt'.format(lan), 'r', encoding=encoding) as f:
		text = f.read()

	words = set(text.split())
	tags = ('<SOL>', '<EOL>')
	words.update(tags)
	word2int = {word:i for i, word in enumerate(words)}
	int2word = dict(enumerate(words))
	print('n_words: {}'.format(len(word2int)))
	print_word2int(lines[:n_lines], word2int)
	return word2int, int2word

def char_encoding():
	with open('en.txt', 'r', encoding='utf-8') as f:
		en_lines = f.readlines()
		# print('num_lines:{}'.format(len(en_lines)))

	with open('fr.txt', 'r', encoding='utf-8-sig') as f:
		fr_lines = f.readlines()
		# print('num_lines:{}'.format(len(fr_lines)))

	input_texts = []
	target_texts = []
	input_chars = set()
	target_chars = set()
	for en_line, fr_line in zip(en_lines, fr_lines):
		input_text = en_line
		target_text = '\t' + fr_line + '\n'
		input_texts.append(en_line) # list of lines
		target_texts.append(target_text)

		for char in input_text:
			if char not in input_chars:
				input_chars.add(char)

		for char in target_text:
			if char not in target_chars:
				target_chars.add(char)

	input_chars = sorted(list(input_chars))
	target_chars = sorted(list(target_chars))
	num_encoder_tokens = len(input_chars)
	num_decoder_tokens = len(target_chars)
	max_encoder_seq_length = max([len(txt) for txt in input_texts])
	max_decoder_seq_length = max([len(txt) for txt in target_texts])
	print('Number of samples:', len(input_texts))
	print('Number of unique input tokens:', num_encoder_tokens)
	print('Number of unique output tokens:', num_decoder_tokens)
	print('Max sequence length for inputs:', max_encoder_seq_length)
	print('Max sequence length for outputs:', max_decoder_seq_length)

	input_token_index = dict(
	    [(char, i) for i, char in enumerate(input_chars)])
	target_token_index = dict(
	    [(char, i) for i, char in enumerate(target_chars)])
	
	encoder_input_data = np.zeros(
	    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
	    dtype='float32')
	decoder_input_data = np.zeros(
	    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
	    dtype='float32')
	decoder_target_data = np.zeros(
	    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
	    dtype='float32')

	for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
	    for t, char in enumerate(input_text):
	        encoder_input_data[i, t, input_token_index[char]] = 1. # one-hot encoding
	    for t, char in enumerate(target_text):
	        # decoder_target_data is ahead of decoder_input_data by one timestep
	        decoder_input_data[i, t, target_token_index[char]] = 1.
	        if t > 0:
	            # decoder_target_data will be ahead by one timestep
	            # and will not include the start character.
	            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

if __name__ == '__main__':
	# n_lines = 1
	# en_word2int, en_int2word = word_encoding(lan='en', n_lines=n_lines)
	# fr_word2int, fr_int2word = word_encoding(lan='fr', n_lines=n_lines)
	char_encoding()