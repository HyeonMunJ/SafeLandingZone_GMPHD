#!/usr/bin/env python

simplesum = sum   # we want to be able to use "pure" sum not numpy (shoulda namespaced)
from numpy import *
import numpy.linalg
from copy import deepcopy
from operator import attrgetter

myfloat = numpy.float64

class GmphdComponent:
	"""Represents a single Gaussian component, 
	with a float weight, vector location, matrix covariance.
	Note that we don't require a GM to sum to 1, since not always about proby densities."""
	def __init__(self, weight, loc, cov):
		self.weight = myfloat(weight)
		self.loc    = array(loc, dtype=myfloat, ndmin=2)
		self.cov    = array(cov, dtype=myfloat, ndmin=2)
		self.loc    = reshape(self.loc, (size(self.loc), 1)) # enforce column vec
		self.cov    = reshape(self.cov, (size(self.loc), size(self.loc))) # ensure shape matches loc shape

		# precalculated values for evaluating gaussian:
		k = len(self.loc)
		self.dmv_part1 = (2.0 * numpy.pi) ** (-k * 0.5)
		self.dmv_part2 = numpy.power(numpy.linalg.det(self.cov), -0.5)
		self.invcov = numpy.linalg.inv(self.cov)

	def dmvnorm(self, x):
		"""Evaluate this multivariate normal component, at a location x.
		NB this does NOT APPLY THE WEIGHTING, simply for API similarity to the other method with this name."""
		x = array(x, dtype=myfloat)
		dev = x - self.loc
		part3 = numpy.exp(-0.5 * dot(dot(dev.T, self.invcov), dev))
		return self.dmv_part1 * self.dmv_part2 * part3

# We don't always have a GmphdComponent object so:
def dmvnorm(loc, cov, x):
	"Evaluate a multivariate normal, given a location (vector) and covariance (matrix) and a position x (vector) at which to evaluate"
	loc = array(loc, dtype=myfloat)
	cov = array(cov, dtype=myfloat)
	x = array(x, dtype=myfloat)
	k = len(loc)
	part1 = (2.0 * numpy.pi) ** (-k * 0.5)
	part2 = numpy.power(numpy.linalg.det(cov), -0.5)
	dev = x - loc
	part3 = numpy.exp(-0.5 * dot(dot(dev.T, numpy.linalg.inv(cov)), dev))

	return part1 * part2 * part3

def sampleGm(complist):
	"Given a list of GmphdComponents, randomly samples a value from the density they represent"
	weights = array([x.weight for x in complist])
	weights = weights / simplesum(weights)   # Weights aren't externally forced to sum to one
	choice = random.random()
	cumulative = 0.0
	for i,w in enumerate(weights):
		cumulative += w
		if choice <= cumulative:
			# Now we sample from the chosen component and return a value
			comp = complist[i]
			return random.multivariate_normal(comp.loc.flat, comp.cov)
	raise RuntimeError("sampleGm terminated without choosing a component")

################################################################################
class Gmphd:
	"""Represents a set of modelling parameters and the latest frame's
		GMM estimate, for a GM-PHD model without spawning.

		Typical usage would be, for each frame of input data, to run:
			g.update(obs)
			g.prune()
			estimate = g.extractstates()

		'gmm' is an array of GmphdComponent items which makes up
			the latest GMM, and updated by the update() call. 
			It is initialised as empty.

			Test code example (1D data, with new trails expected at around 100):
		from gmphd import *
		g = Gmphd([GmphdComponent(1, [100], [[10]])], 0.9, 0.9, [[1]], [[1]], [[1]], [[1]], 0.000002)
		g.update([[30], [67.5]])
		g.gmmplot1d()
		g.prune()
		g.gmmplot1d()

		g.gmm

		[(float(comp.loc), comp.weight) for comp in g.gmm]
	"""
	
	def __init__(self, survival, detection, f, q, h, r, clutter, birthgmm, h_star):
		"""
		  'birthgmm' is an array of GmphdComponengo!
		  'f' is state transition matrix F.
		  'q' is the process noise covariance Q.
		  'h' is the observation matrix H.
		  'r' is the observation noise covariance R.
		  'clutter' is the clutter intensity.
		  """
		# self.gmm = []  # empty - things will need to be born before we observe them
		self.birthgmm = birthgmm
		self.gmm = deepcopy(self.birthgmm)
		# self.gmm = []
		self.survival = myfloat(survival)        # p_{s,k}(x) in paper
		self.detection = myfloat(detection)      # p_{d,k}(x) in paper
		self.f = array(f, dtype=myfloat)   # state transition matrix      (F_k-1 in paper)
		self.q = array(q, dtype=myfloat)   # process noise covariance     (Q_k-1 in paper)
		self.h = array(h, dtype=myfloat)   # observation matrix           (H_k in paper)
		self.r = array(r, dtype=myfloat)   # observation noise covariance (R_k in paper)
		self.clutter = myfloat(clutter)   # clutter intensity (KAU in paper)

		self.h_star = array(h_star, dtype=myfloat)

		self.edge = [0., 0., 0., 0.]

		self.weight_A = 5e-1
		self.weight_B = 1

		self.flag_meas_update = False
		
	def meas_classification(self, obs_set):
		# considering the mahalanobis distance between measurements and predicted target states,
		# classify measurements into meas from birth targets and meas from surviving targets
		obs_b = []
		obs_s = []
		T_s = 3 # threshold 

		for obs in obs_set:
			flag_inside = False
			for comp in self.gmm:
				hf_loc = dot(dot(self.h, self.f), comp.loc)
				diff = linalg.norm([obs[0] - hf_loc[0], obs[1] - hf_loc[1], obs[2] - hf_loc[2]])
				# print('diff = ', diff)
				if diff < T_s:
					obs_s.append(obs)
					flag_inside = True
					break
			if self.gmm == []:
				obs_s.append(obs)
			elif not flag_inside:
				obs_b.append(obs)

		obs_A = []
		obs_B = []
		for obs in obs_b:
			if self.edge[0] <= obs[0] <= self.edge[1] and self.edge[2] <= obs[1] <= self.edge[3]:
				obs_A.append(obs)
			else:
				obs_B.append(obs)

		return obs_A, obs_B, obs_s

	def update(self, obs, f, vel):
		"""Run a single GM-PHD step given a new frame of observations.
		  'obs' is an array (a set) of this frame's observations.
		  Based on Table 1 from Vo and Ma paper."""
		self.f = array(f, dtype=myfloat)

		if array(obs).ndim == 1:
			obs = [obs]

		#######################################
		# Step 1 - prediction for existing targets

		updated = [GmphdComponent(                        \
					self.survival * comp.weight,          \
					dot(self.f, comp.loc),                \
					self.q + dot(dot(self.f, comp.cov), self.f.T)   \
			) for comp in self.gmm]
	
		# predicted = born + spawned + updated
		predicted = updated

		######################################
		# Step 2 - measurement classification
		obs_A, obs_B, obs_surv = self.meas_classification(obs)		

		print("size of obs_surv : ", size(obs_surv)/6, 'size of obs_A : ', size(obs_A)/6, 'size of obs_B : ', size(obs_B)/6)

		#######################################
		# Step 3 - construction of PHD update components
		# These two are the mean and covariance of the expected observation
		nu = [dot(self.h, comp.loc)                         for comp in predicted]
		s  = [self.r + dot(dot(self.h, comp.cov), self.h.T) for comp in predicted]
		# Not sure about any physical interpretation of these two...
		k = [dot(dot(comp.cov, self.h.T), linalg.inv(s[index]))
						for index, comp in enumerate(predicted)]
		pkk = [dot(eye(len(k[index])) - dot(k[index], self.h), comp.cov)
						for index, comp in enumerate(predicted)]

		pkk_b = pkk # should be modified properly

		#######################################
		# Step 4 - update states of surviving targets using observations
		# The 'predicted' components are kept, with a decay
		newgmm = [GmphdComponent(comp.weight * (1.0 - self.detection), comp.loc, comp.cov) for comp in predicted]

		for anobs in obs:
			anobs_arr = array(anobs)
			anobs_arr = reshape(anobs, (size(anobs),1))
			newgmmpartial = []
			for j, comp in enumerate(predicted):
				newgmmpartial.append(GmphdComponent(          \
						self.detection * comp.weight          \
							* dmvnorm(nu[j], s[j], anobs_arr),    \
						comp.loc + dot(k[j], anobs_arr - nu[j]),  \
						pkk[j]                                \
						))
	
			# The Kappa thing (clutter and reweight)
			weightsum = simplesum(newcomp.weight for newcomp in newgmmpartial)

			arr_surv = array(obs_surv)
			arr_A = array(obs_A)
			arr_B = array(obs_B)
			if anobs in arr_surv:
				reweighter = 1.0 / (self.clutter + weightsum)

			elif anobs in arr_A:
				ele_vel = [0, -vel[0], 0, -vel[1], 0, -vel[2], 0, 0, 0]
				ele_vel = reshape(array(ele_vel), (9,1))
				newgmmpartial.append(GmphdComponent(          \
						self.weight_A,    \
						dot(self.h_star, anobs_arr) + ele_vel, # should be modified along the dimension of the state \ 
						pkk_b[j]                                \
						))
				reweighter = 1.0 / (self.clutter + weightsum + self.weight_A)

			elif anobs in arr_B:
				ele_vel = [0, -vel[0], 0, -vel[1], 0, -vel[2], 0, 0, 0]
				ele_vel = reshape(array(ele_vel), (9,1))
				newgmmpartial.append(GmphdComponent(          \
						self.weight_B,    \
						dot(self.h_star, anobs_arr) + ele_vel, # should be modified along the dimension of the state \ 
						pkk_b[j]                                \
						))
				reweighter = 1.0 / (self.clutter + weightsum + self.weight_B)

			else:
				print('failed: ', anobs)

			for newcomp in newgmmpartial:
				newcomp.weight *= reweighter

			newgmm.extend(newgmmpartial)

		self.gmm = newgmm

	def prune(self, truncthresh=1e-20, mergethresh=0.1, maxcomponents=100):
		"""Prune the GMM. Alters model state.
		  Based on Table 2 from Vo and Ma paper."""
		# Truncation is easy
		weightsums = [simplesum(comp.weight for comp in self.gmm)]   # diagnostic
		sourcegmm = [comp for comp in self.gmm if comp.weight > truncthresh]
		weightsums.append(simplesum(comp.weight for comp in sourcegmm))
		origlen  = len(self.gmm)
		trunclen = len(sourcegmm)
		# Iterate to build the new GMM
		newgmm = []
		while len(sourcegmm) > 0:
			# find weightiest old component and pull it out
			windex = argmax(comp.weight for comp in sourcegmm)
			weightiest = sourcegmm[windex]
			sourcegmm = sourcegmm[:windex] + sourcegmm[windex+1:]
			# find all nearby ones and pull them out
			distances = [float(dot(dot((comp.loc - weightiest.loc).T, comp.invcov), comp.loc - weightiest.loc)) for comp in sourcegmm]
			dosubsume = array([dist <= mergethresh for dist in distances])
			subsumed = [weightiest]
			if any(dosubsume):
				#print "Subsuming the following locations into weightest with loc %s and weight %g (cov %s):" \
				#	% (','.join([str(x) for x in weightiest.loc.flat]), weightiest.weight, ','.join([str(x) for x in weightiest.cov.flat]))
				#print list([comp.loc[0][0] for comp in list(array(sourcegmm)[ dosubsume]) ])
				subsumed.extend( list(array(sourcegmm)[ dosubsume]) )
				sourcegmm = list(array(sourcegmm)[~dosubsume])
			# create unified new component from subsumed ones
			aggweight = simplesum(comp.weight for comp in subsumed)
			newcomp = GmphdComponent( \
				aggweight,
				sum(array([comp.weight * comp.loc for comp in subsumed]), 0) / aggweight,
				sum(array([comp.weight * (comp.cov + (weightiest.loc - comp.loc) \
							* (weightiest.loc - comp.loc).T) for comp in subsumed]), 0) / aggweight
					)
			newgmm.append(newcomp)

		# Now ensure the number of components is within the limit, keeping the weightiest
		newgmm.sort(key=attrgetter('weight'))
		newgmm.reverse()
		self.gmm = newgmm[:maxcomponents] # what if less than maxcomp?
		weightsums.append(simplesum(comp.weight for comp in newgmm))
		weightsums.append(simplesum(comp.weight for comp in self.gmm))
		print("prune(): %i -> %i -> %i -> %i" % (origlen, trunclen, len(newgmm), len(self.gmm)))
		print("prune(): weightsums %g -> %g -> %g -> %g" % (weightsums[0], weightsums[1], weightsums[2], weightsums[3]))
		# pruning should not alter the total weightsum (which relates to total num items) - so we renormalise
		if weightsums[3] == 0:
			weightnorm = 0
			print('weightsum[3] is 0!! I think something is going to wrong way!!!')
		else:
			weightnorm = weightsums[0] / weightsums[3]
		for comp in self.gmm:
			comp.weight *= weightnorm

	def extractstatesmax(self, bias = 1.0):
		items = []
		print("weights:")
		print([round(comp.weight, 10) for comp in self.gmm])
		max_val = -1.
		for comp in self.gmm:
			val = comp.weight * float(bias)
			if val > max_val:
				max_val = val
				items = deepcopy(comp.loc)
		return items, max_val
		
'''
	def extractstates(self, bias=1.0):
		"""Extract the multiple-target states from the GMM.
		  Returns a list of target states; doesn't alter model state.
		  Based on Table 3 from Vo and Ma paper.
		  I added the 'bias' factor, by analogy with the other method below."""
		items = []
		print("weights:")
		print([round(comp.weight, 7) for comp in self.gmm])
		for comp in self.gmm:
			val = comp.weight * float(bias)
			if val > 0.5:
				for _ in range(int(round(val))):
					items.append(deepcopy(comp.loc))
		for x in items: print(x.T)
		return items

	def extractstatesusingintegral(self, bias=1.0):
		"""Extract states based on the expected number of states from the integral of the intensity.
		This is NOT in the GMPHD paper; added by Dan.
		"bias" is a multiplier for the est number of items.
		"""
		numtoadd = int(round(float(bias) * simplesum(comp.weight for comp in self.gmm)))
		print("bias is %g, numtoadd is %i" % (bias, numtoadd))
		items = []
		# A temporary list of peaks which will gradually be decimated as we steal from its highest peaks
		peaks = [{'loc':comp.loc, 'weight':comp.weight} for comp in self.gmm]
		while numtoadd > 0:
			windex = 0
			wsize = 0
			for which, peak in enumerate(peaks):
				if peak['weight'] > wsize:
					windex = which
					wsize = peak['weight']
			# add the winner
			items.append(deepcopy(peaks[windex]['loc']))
			peaks[windex]['weight'] -= 1.0
			numtoadd -= 1
		for x in items: print(x.T)
		return items

	########################################################################################
	def gmmeval(self, points, onlydims=None):
		"""Evaluates the GMM at a supplied list of points (full dimensionality). 
		'onlydims' if not nil, marginalises out (well, ignores) the nonlisted dims. All dims must still be listed in the points, so put zeroes in."""
		return [ \
			simplesum(comp.weight * comp.dmvnorm(p) for comp in self.gmm) \
				for p in points]
	def gmmeval1d(self, points, whichdim=0):
		"Evaluates the GMM at a supplied list of points (1D only)"
		return [ \
			simplesum(comp.weight * dmvnorm([comp.loc[whichdim]], [[comp.cov[whichdim][whichdim]]], p) for comp in self.gmm) \
				for p in points]

	def gmmevalgrid1d(self, span=None, gridsize=200, whichdim=0):
		"Evaluates the GMM on a uniformly-space grid of points (1D only)"
		if span==None:
			locs = array([comp.loc[whichdim] for comp in self.gmm])
			span = (min(locs), max(locs))
		grid = (arange(gridsize, dtype=float) / (gridsize-1)) * (span[1] - span[0]) + span[0]
		return self.gmmeval1d(grid, whichdim)


	def gmmevalalongline(self, span=None, gridsize=200, onlydims=None):
		"""Evaluates the GMM on a uniformly-spaced line of points (i.e. a 1D line, though can be angled).
		'span' must be a list of (min, max) for each dimension, over which the line will iterate.
		'onlydims' if not nil, marginalises out (well, ignores) the nonlisted dims. All dims must still be listed in the spans, so put zeroes in."""
		if span==None:
			locs = array([comp.loc for comp in self.gmm]).T   # note transpose - locs not a list of locations but a list of dimensions
			span = array([ map(min,locs), map(max,locs) ]).T   # note transpose - span is an array of (min, max) for each dim
		else:
			span = array(span)
		steps = (arange(gridsize, dtype=float) / (gridsize-1))
		grid = array(map(lambda aspan: steps * (aspan[1] - aspan[0]) + aspan[0], span)).T  # transpose back to list of state-space points
		return self.gmmeval(grid, onlydims)

	def gmmplot1d(self, gridsize=200, span=None, obsnmatrix=None):
		"Plots the GMM. Only works for 1D model."
		import matplotlib.pyplot as plt
		vals = self.gmmevalgrid1d(span, gridsize, obsnmatrix)
		fig = plt.figure()
		# plt.plot(grid, vals, '-')
		fig.show()
		return fig
'''