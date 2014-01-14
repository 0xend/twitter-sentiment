#!/usr/bin/env python

import sys
import re
import os
import resource
import random
from utilities import *
resource.setrlimit(resource.RLIMIT_NOFILE, (10000,-1))

DATE = 'T'
BODY = 'W'
USER = 'U'
STOP_WORDS = 'stopwords.txt'
ABBR = 'abbreviations.txt'
NO_CONTENT =  'No Post Title'
BUF_SIZE = 4096

class Parser:
	
	
	def __init__(self, out_dir, in_dir, version):
		self.OUT_DIR = out_dir
		self.IN_DIR = in_dir
		self.version = version
		self.files = get_files(in_dir)
		self.stop_words = self._fill_sw()	
		self.abbr = self._fill_abbreviations()

	def _fill_sw(self):
		f = open(STOP_WORDS)
		sw = set()
		for word in f:
			sw.add(word[:-1])
		return sw

	def _fill_abbreviations(self):
		f = open(ABBR)
		abbr = {}
		for line in f:
			parts = line.split(';')
			abbr[parts[0]] = parts[1][:-2]
		return abbr

	def _multiple_replace(self, text):
		regex_str ="\\b|\\b".join(map(re.escape, self.abbr.keys())) 
		regex_str = "\\b" + regex_str + "\\b"
		regex = re.compile(regex_str)
		return regex.sub(lambda mo: self.abbr[mo.group(0)], text) 


	# Parses the content of each tweet
	def extract_data(self, line, sw):
		REPLACEMENTS = r'http://[\w+\.\/]+|@\w+'
		SW = r'\b(' + r'|'.join(self.stop_words) + r')\b\s*'
		content_regex = re.compile(r'%c\t(.*)' % BODY)
		content = content_regex.match(line).group(1)
		stripped = re.sub(REPLACEMENTS, '', content).strip()
		if sw:
			stripped = re.sub(SW, '', stripped)
		
		if stripped.strip() == NO_CONTENT or len(stripped) == 0:
			return None
		else:
			return stripped

	def parse_file(self,fname, sw):
		f = open(fname)
		open_files = {}
		date = ''
		for line in f:
			if line.startswith(DATE):
				parts = line.split()
				dates = parts[1].split('-')
				date = '%s-%s' % (parts[1], parts[2][0:2])
				if not date in open_files:
					open_files[date] = open('%s/%s.txt' % (self.OUT_DIR, date), 'a', BUF_SIZE)
				
			elif line.startswith(BODY):
				data = self.extract_data(line,sw)
				if data is not None:	
					open_files[date].write(data+ '\n') 
		
		for pointer in open_files.values():
			pointer.close()
	
	def parse_second(self,fname):
		DIVISOR = '---'
		WORD_MIN = 3
		f = open('%s/%s' % (self.IN_DIR, fname))
		sent_feed = open('%s/sentiment_unparsed_%s/%s' % (self.OUT_DIR, self.version, fname), 'a', BUF_SIZE)
		ready = open('%s/sentiment_ready_%s/%s' % (self.OUT_DIR, self.version, fname), 'a', BUF_SIZE)
		emoticons = load_emoticons()
		re_pos = re.compile(emoticons[0])
		re_neg = re.compile(emoticons[1])
		for tweet in f:
			tweet = re.sub('RT', '', tweet)
			tweet = tweet.strip()
			tweet = self._multiple_replace(tweet)
			is_pos = re_pos.match(tweet)
			is_neg = re_neg.match(tweet)
			
			PUNCT = r'[^a-zA-Z0-9\.\-\#]'
			tweet = re.sub(PUNCT, ' ', tweet)
			tweet = re.sub('\.+', '.', tweet)
			tweet = re.sub('\ +', ' ', tweet)
			
			if is_pos and is_neg:
				continue
			if is_pos:
				ready.write('%s|;;|1\n' % tweet)
			elif is_neg:
				ready.write('%s|;;|-1\n' % tweet)
			else:
				if len(tweet) > WORD_MIN:
					if tweet[len(tweet)-1] != '.':
						tweet += '.'
					sent_feed.write('%s\n%s\n' % (tweet, DIVISOR))
		f.close()
		ready.close()
		sent_feed.close()

	def parse_files(self, type_parser):
		sw = False if type_parser != 1 else True
		for fname in self.files:
			print 'Analyzing %s...' % fname
			if type_parser  < 2:
				self.parse_file('%s/%s' % (self.IN_DIR, fname), sw)
			elif type_parser == 2:
				self.parse_second(fname)
	
def main():
	dirname_output= sys.argv[1]
	dirname_input = sys.argv[2]
	type_parser = int(sys.argv[3])
	version = sys.argv[4]
	parser = Parser(dirname_output, dirname_input, version)
	parser.parse_files(type_parser)

if __name__ == '__main__':
	main()
