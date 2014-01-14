from HashGraph import *

NEG=0
POS=1

class LBP():
	def __init__(self, g):
		self.graph = g

	def _init_msgs(self):
		self.m = {}
		for n1, neighbors in self.graph.edges.items():
			for n2 in neighbors:
				self.m[(n1,n2)] = [1, 1]

	def _potential_ind(self, h, sent):
		reg = self.graph.nodes[h][1]
		return (1+reg)/2 if sent == POS else (1-reg)/2 
	
	def _potential_pair(self, h1, h2, y_i, y_j):
		if y_i == y_j:
			coeff = float(self.graph.weights[(h1,h2)]) / (self.graph.nodes[h1][0]+self.graph.nodes[h2][0])
		else:
			coeff = 0
		return coeff
	
	def _update(self, h_i):
		max_dif = 0
		extra = 0.0000001
		for h_j in self.graph.edges[h_i]:
			orig = []
			for i in [NEG,POS]:
				orig.append(self.m[(h_i, h_j)][i])

			for j in [NEG,POS]:
				result = 0
				for i in [NEG,POS]:
					term_1 = self._potential_pair(h_i, h_j, i, j) * self._potential_ind(h_i, i)
					term_2 = 1
					aux = []
					for h_k in self.graph.edges[h_i]:
						if h_k == h_j:
							continue
						term_2 *= self.m[(h_k, h_i)][i]
						aux.append(self.m[(h_k, h_i)][i])
					result += term_1 * term_2
				self.m[(h_i, h_j)][j] = result + extra	
			#normalize
			tot = sum(self.m[(h_i,h_j)])

			for i in [NEG, POS]:
				self.m[(h_i,h_j)][i] /= tot
			new = self.m[(h_i, h_j)]
			max_dif = max(abs(new[POS]-orig[POS]), abs(new[NEG]-orig[NEG]))
		
		return max_dif

	def _update_m(self):
		self._init_msgs()
		delta = 0.001
		max_dif = 0.001
		run = 0
		print 'Running LBP...'
		while delta <= max_dif:
			max_dif = 0.0
			for h_i in self.graph.nodes:
				dif = self._update(h_i)
				max_dif = max(dif, max_dif)
			print run, max_dif
			run += 1
		

	def final_values(self):
		self._update_m()
		results = {}
		count = 0
		for h_i in self.graph.nodes:
			h_i_prob = []
			for i in [NEG,POS]:
				res = 1
				for h_j in self.graph.edges[h_i]:
					res *= self.m[(h_j, h_i)][i]
				res *= self._potential_ind(h_i, i)
				h_i_prob.append(res)
			tot = sum(h_i_prob)
			if tot == 0: 
				continue
			for i in [NEG,POS]:
				h_i_prob[i] /= tot
			results[h_i] = h_i_prob
			if h_i_prob[0] >= .5:
				count += 1
		return results



