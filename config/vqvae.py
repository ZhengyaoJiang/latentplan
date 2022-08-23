from latentplan.utils import watch

#------------------------ base ------------------------#

logbase = '~/logs/'
gpt_expname = 'vae/vq'

## automatically make experiment names for planning
## by labelling folders with these args
args_to_watch = [
    ('prefix', ''),
    ('plan_freq', 'freq'),
    ('horizon', 'H'),
    ('beam_width', 'beam'),
]

base = {

    'train': {
        'model': "VQTransformer",
        'tag': "experiment",
        'state_conditional': True,
        'N': 100,
        'discount': 0.99,
        'n_layer': 4,
        'n_head': 4,

        ## number of epochs for a 1M-size dataset; n_epochs = 1M / dataset_size * n_epochs_ref
        'n_epochs_ref': 50,
        'n_saves': 3,
        'logbase': logbase,
        'device': 'cuda',

        'K': 512,
        'latent_step': 3,
        'n_embd': 128,
        'trajectory_embd': 512,
        'batch_size': 512,
        'learning_rate': 2e-4,
        'lr_decay': False,
        'seed': 42,

        'embd_pdrop': 0.1,
        'resid_pdrop': 0.1,
        'attn_pdrop': 0.1,

        'step': 1,
        'subsampled_sequence_length': 25,
        'termination_penalty': -100,
        'exp_name': gpt_expname,

        'position_weight': 1,
        'action_weight': 5,
        'reward_weight': 1,
        'value_weight': 1,

        'first_action_weight': 0,
        'sum_reward_weight': 0,
        'last_value_weight': 0,
        'suffix': '',

        "normalize": True,
        "normalize_reward": True,
        "max_path_length": 1000,
        "bottleneck": "pooling",
        "masking": "uniform",
        "disable_goal": False,
        "residual": True,
        "ma_update": True,
    },

    'plan': {
        'discrete': False,
        'logbase': logbase,
        'gpt_loadpath': gpt_expname,
        'gpt_epoch': 'latest',
        'device': 'cuda',
        'renderer': 'Renderer',
        'suffix': '0',

        'plan_freq': 1,
        'horizon': 15,

        "rounds": 2,
        "nb_samples": 4096,

        'beam_width': 64,
        'n_expand': 4,

        'prob_threshold': 0.05,
        'prob_weight': 5e2,

        'vis_freq': 200,
        'exp_name': watch(args_to_watch),
        'verbose': True,
        'uniform': False,

        # Planner
        'test_planner': 'beam_prior',
    },

}

#------------------------ locomotion ------------------------#

hammer_cloned_v0 = hammer_human_v0 = human_expert_v0 = {
    'train': {
        "termination_penalty": None,
        "max_path_length": 200,
        'n_epochs_ref': 10,
        'subsampled_sequence_length': 25,
    },
    'plan': {
        'horizon': 24,
    }
}

relocate_cloned_v0 = relocate_human_v0 = relocate_expert_v0 = {
    'train': {
        "termination_penalty": None,
        "max_path_length": 200,
        'n_epochs_ref': 10,
        'subsampled_sequence_length': 25,
    },
    'plan': {
        'horizon': 24,
    }
}

door_cloned_v0 = door_human_v0 = door_expert_v0 = {
    'train': {
        "termination_penalty": None,
        "max_path_length": 200,
        'n_epochs_ref': 10,
        'subsampled_sequence_length': 25,
    },
    'plan': {
        'horizon': 24,
    }
}

pen_cloned_v0 = pen_expert_v0 = pen_human_v0 = {
    'train': {
        "termination_penalty": None,
        "max_path_length": 100,
        'n_epochs_ref': 10,
        'subsampled_sequence_length': 25,
    },
    'plan': {
        'prob_weight': 5e2,
        'horizon': 24,
    }
}

antmaze_large_diverse_v0=antmaze_large_play_v0=antmaze_medium_diverse_v0=antmaze_medium_play_v0=antmaze_umaze_v0 = {
    'train':{
        "disable_goal": False,
        "termination_penalty": None,
        "max_path_length": 1001,
        "normalize": False,
        "normalize_reward": False,
        'lr_decay': False,
        'K': 8192,
        "discount": 0.998,
        'subsampled_sequence_length': 16,
    },
    'plan': {
        'iql_value': False,
        'horizon': 15,
        'vis_freq': 200,
        'renderer': "AntMazeRenderer"
    }
}

antmaze_ultra_diverse_v0=antmaze_ultra_play_v0 = {
'train':{
        "disable_goal": False,
        "termination_penalty": None,
        "max_path_length": 1001,
        "normalize": False,
        "normalize_reward": False,
        'n_epochs_ref': 50,
        'lr_decay': False,
        'K': 8192,
        "discount": 0.998,
        'batch_size': 512,
        'subsampled_sequence_length': 16,
    },
    'plan': {
        'iql_value': False,
        'horizon': 15,
        'vis_freq': 200,
        'renderer': "AntMazeRenderer"
    }
}