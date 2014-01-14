from utilities import *
import sys 
import re
import itertools
import LBP


class Graph():
	
	def __init__(self):
		self.nodes = {}
		self.edges = {}
		self.weights = {}

	def __str__(self):
		return 'Nodes: %d, Edges: %d' % len(self.nodes, self.edges/2)

	def add_node(self, value, sentiment=0):
		if value in self.nodes:
			self.nodes[value] = (self.nodes[value][0]+1, 
				(self.nodes[value][1] + sentiment)/(self.nodes[value][0]+1)
				)
		else:
			self.nodes[value] = (1, sentiment)
			self.edges[value] = set()

	def size(self):
		return len(self.nodes)
	
	def total_edges(self):
		return len(self.edges)/2

	def add_edge(self, n1, n2):
		self._add_edge(n1, n2)
		self._add_edge(n2, n1)

	def _add_edge(self, n1, n2):
		self.edges.setdefault(n1, set())
		self.edges[n1].add(n2)
		if (n1,n2) in self.weights:
			self.weights[(n1,n2)] += 1
		else:
			self.weights[(n1,n2)] = 1
	
class HashGraph():

	def __init__(self, files):
		self.files = files
		self.graph = Graph()


	def _generate_from_file(self, fname):
		f = open(fname)
		ht = re.compile('(#\w+)')
		hts = [] 
		hts_set = set()
		for line in f:
			hashtags = set(ht.findall(line))
			sent = line.split('|;;|')[1]
			for h in hashtags:
				self.graph.add_node(h, float(sent))
			for pair in itertools.combinations(hashtags, r=2):
				self.graph.add_edge(pair[0], pair[1])

		f.close()

	def generate_graph(self):
		for f in self.files:
			self._generate_from_file(f)


