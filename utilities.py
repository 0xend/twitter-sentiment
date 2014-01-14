import os, re
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords

def is_english(text):
	tokens = wordpunct_tokenize(text.lower())
	percentage = {}
	langs = ['english', 'spanish', 'portuguese', 'russian', 'french']
	for lang in langs:
		percentage[lang] = len(set(tokens) & set(stopwords.words(lang)))
	most = sorted(percentage, key = percentage.get, reverse=True)[0]
	return most == 'english'

def load_afinn(fname):
	f = open(fname)
	word_sents = {}
	bigrams = []
	for line in f:
		parts = line.split()
		if len(parts) > 2:
			bigrams.append(' '. join(parts[:-1]))
		word_sents[' '.join(parts[:-1])] = int(parts[len(parts)-1])
	return (word_sents, bigrams)

# Returns a list with the txt files in a dir
def get_files(dirname):
	all_files = os.listdir(dirname)	
	data = []
	for fname in all_files:
		data.append(fname)
	return data
def load_emoticons():
	f = open('emoticons.txt')
	pos = f.readline().strip()
	neg = f.readline().strip()
		
	items_pos = pos.split()
	items_neg = neg.split()

	escape_pos = [ re.escape(v) for v in items_pos ]
	escape_neg = [ re.escape(v) for v in items_neg ]
		
	re_pos = r'.*(' + r'|'.join(escape_pos) + r').*'
	re_neg = r'.*(' + r'|'.join(escape_neg) + r').*'

	return (re_pos, re_neg)	

def avg(l):
	return sum(l)/len(l)
