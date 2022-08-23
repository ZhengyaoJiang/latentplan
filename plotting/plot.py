import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pdb


class Colors:
	grey = '#B4B4B4'
	gold = '#F6C781'
	red = '#EC7C7D'
	blue = '#70ABCC'
	green = '#7CB5A1'

LABELS = {
	'TAP': 'Full Model',
	'$L=1$': 'Latent Steps',
	'$L=4$': 'Latent Steps',
	'$\\beta=10^{-5}$': 'Likelihood Threshold',
	'\\beta=0.5': 'Likelihood Threshold',
	'Horizon=3': 'Planning Horizon',
	'Horizon=21': 'Planning Horizon'
}

def get_mean(results, exclude=None):
	'''
		results : { environment: score, ... }
	'''
	filtered = {
		k: v for k, v in results.items()
		if (not exclude) or (exclude and exclude not in k)
	}
	return np.mean(list(filtered.values()))

if __name__ == '__main__':

	#################
	## latex
	#################
	matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
	matplotlib.rc('text', usetex=True)
	matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
	#################

	fig = plt.gcf()
	ax = plt.gca()
	fig.set_size_inches(9.5, 3)

	means = {'TAP': 82.5, '$L=1$': 60.6, '$L=4$':82.1, r'$\beta=10^{-5}$':57.6, r'$\beta=0.5$':65.8,
			 'Horizon=3': 71.9, 'Horizon=21':82.4, 'Prior': 78.3, 'Uniform': 51.3}

	algs = ['TAP', '$L=1$', '$L=4$', r'$\beta=10^{-5}$', r'$\beta=0.5$', 'Horizon=3', 'Horizon=21',
			'Prior', 'Uniform']
	vals = [means[alg] for alg in algs]

	colors = [
		Colors.grey, Colors.gold,
		Colors.gold, Colors.red, Colors.red, Colors.blue, Colors.blue, Colors.green, Colors.green
	]

	labels = [alg for alg in algs]
	plt.bar(labels, vals, color=colors, edgecolor=Colors.green, lw=0)
	plt.ylabel('Mean normalized return', labelpad=15)

	legend_labels = ['Full Model', 'Latent Steps', 'Likelihood Threshold', 'Planning Horizon', 'Direct Sampling']
	colors = [Colors.grey, Colors.gold, Colors.red, Colors.blue, Colors.green]
	handles = [plt.Rectangle((0,0),1,1, color=color) for label, color in zip(legend_labels, colors)]
	plt.legend(handles, legend_labels, ncol=5,
		bbox_to_anchor=(1.01, -.18), fancybox=False, framealpha=0, shadow=False, columnspacing=1.5, handlelength=1.5)

	matplotlib.rcParams['hatch.linewidth'] = 7.5
	ax.patches[-1].set_hatch('/')

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	plt.savefig('ablationbar.png', bbox_inches='tight', dpi=500)
