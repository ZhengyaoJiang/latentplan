import latentplan.utils as utils
from os.path import join
import latentplan.datasets as datasets
import os

class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-expert-v2'
    #config: str = 'config.offline_continuous'
    config: str = 'config.vqvae'

#######################
######## setup ########
#######################

args = Parser().parse_args('plan')
args.nb_samples = int(args.nb_samples)
args.n_expand = int(args.n_expand)
args.beam_width = int(args.beam_width)
args.rounds = int(args.rounds)
args.logbase = os.path.expanduser(args.logbase)
args.savepath = os.path.expanduser("tmp/visualizedata")
args.uniform = bool(args.uniform)
try:
    args.prob_weight = float(args.prob_weight)
except:
    args.prob_weight = 5e2


#######################
####### models ########
#######################

dataset = utils.load_from_config(args.logbase, args.dataset, args.exp_name,
        'data_config.pkl')
rollout = dataset.joined_segmented[0]

renderer = utils.make_renderer(args)
renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)
