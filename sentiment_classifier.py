from utilities import *
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.linear_model import RidgeClassifier, Ridge, SGDRegressor, RidgeCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn import metrics
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import random
import re
import sys
import numpy as np

class SentimentClassifier():
	
	def __init__(self, dirname):
		self.data_files	= get_files(dirname)
		self.pos = []
		self.neg = []
		emoticons = load_emoticons()
		for fname in self.data_files:
			f = open('%s/%s' % (dirname, fname))
			for line in f:
				tweet, sent = line.split('|;;|')	
				tweet = tweet.replace('#', ' ')
				sent = int(sent[:-1])
				if sent == 1:
					tweet = re.sub(emoticons[0][3:-3], ' ', tweet)
					self.pos.append(tweet)
				else:
					tweet = re.sub(emoticons[1][3:-3], ' ', tweet)
					self.neg.append(tweet)
		random.shuffle(self.pos)
		random.shuffle(self.neg)
		self.hv = HashingVectorizer(stop_words='english', strip_accents='ascii', non_negative=True)
		
	def mix_data(self, perc, label = 2):
		ratio = float(len(pos)) / len(neg) 
		train_pos = int(len(pos)  * (perc/100.0))
		train_neg = int(len(neg) * (perc/100.0))
	
		test_samples = len(neg) - train_neg

		indeces = range(train_pos+train_neg)
		random.shuffle(indeces)
		X_train = []
		y_train = []
		for i in indeces:
			if i >= train_pos:
				X_train.append(neg[i-train_pos])
				y_train.append(-1)
			else:
				X_train.append(pos[i])
				y_train.append(1)

			X_train = hv.transform(X_train)
			y_train = np.array(y_train)

		
		if label == 0:
			X_test = hv.transform(neg[train_neg:len(neg)])
			y_test = [-1] * (len(neg) - train_neg)
		elif label == 1:
			X_test = hv.transform(pos[train_pos:len(pos)])
			y_test = [1] * (len(pos) - train_pos)
		else:
			X_test = hv.transform(pos[train_pos:train_pos+test_samples] + neg[train_neg:len(neg)])
			y_test = [1] * (test_samples) + [-1] * (test_samples)

		y_test = np.array(y_test)

		return X_train, y_train, X_test, y_test


	def reg_to_class(self, y_test,values, cut):
		new_values = []
		y_test_new = []
		for i in range(0, len(values)):
			if values[i] < cut[0]:	
				new_values.append(-1.)
				y_test_new.append(y_test[i])
			elif values[i] > cut[1]: 
				new_values.append(1.)
				y_test_new.append(y_test[i])
		data_kept = float(len(new_values)) / values.size
		return (y_test_new, new_values, data_kept)
	
	def discriminate_precision(self, y_test, pred):
		positives = 0
		negatives = 0
		right_positives = 0
		right_negatives = 0
		for i in range(0, len(y_test)):
			if y_test[i] == 1:
				positives += 1
				if pred[i] == 1:
					right_positives += 1
			else:
				negatives += 1
				if pred[i] == 0:
					right_negatives += 1
		return (float(right_positives)/positives, float(right_negatives)/negatives)

	def train(self, X_train, y_train, clfs):
		self.clfs = clfs
		for clf in clfs:
			clf.fit(X_train, y_train)
		
	def test(self, X_test, y_test,clfs_arg=None):
		clfs = clfs_arg if clfs_arg is not None else self.clfs
		for clf in clfs:
			pred = clf.predict(X_test)
			data_kept = 1.
			if not classifier:
				#cuts = [(0.3, 0.7), (0.35, 0.65), (0.4, 0.6), (0.45, 0.55) (0.5,0.5)]
				#for different cuts
				(y_test, pred, data_kept) = reg_to_class(y_test, pred, (-.2,.2))
		
			score = metrics.f1_score(y_test, pred)
			discriminate_label(y_test, pred)
		return score, data_kept
	
	def swipe_clf(c, arg, values, X, y,classifier=True, *args,  **kwargs):
		max_score = 0
		best_arg = values[0]
		args = {}
		max_keep = 1.
		best_clf = None
		kfold = KFold(n=len(y), n_folds=10, indices=True)
		for k,v in kwargs.items():
			args[k] = v
		for v in values:
			args[arg] = v
			clf = c(**args)
			scores = []	
			keep = []
			for train, test in kfold:
				result, data_kept = get_results(X[train], y[train], X[test], y[test],clf, classifier)
				keep.append(data_kept)
				scores.append(result)
			score = avg(scores)
			if score > max_score:
				best_clf = clf
				max_score = score
				best_arg = v
				max_keep = avg(keep)
		return (best_arg, max_score, max_keep, best_clf)

def main():
	dirname = sys.argv[1]
	osc = SentimentClassifier(dirname)

if __name__ == '__main__':
	main()
