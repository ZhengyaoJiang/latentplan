import os
import importlib
import random
import numpy as np
import torch
import pdb

from tap import Tap
from .serialization import mkdir
from .arrays import set_device
from .git_utils import (
    get_git_rev,
    save_git_diff,
)

def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def watch(args_to_watch):
    def _fn(args):
        exp_name = []
        for key, label in args_to_watch:
            if not hasattr(args, key):
                continue
            val = getattr(args, key)
            exp_name.append(f'{label}{val}')
        exp_name = '_'.join(exp_name)
        exp_name = exp_name.replace('/_', '/')
        return exp_name
    return _fn

class Parser(Tap):
    def save(self):
        fullpath = os.path.join(os.path.expanduser(self.savepath), 'args.json')
        print(f'[ utils/setup ] Saved args to {fullpath}')
        #super().save(fullpath, skip_unpicklable=True)

    def parse_args(self, experiment=None):
        args = super().parse_args(known_only=True)
        ## if not loading from a config script, skip the result of the setup
        if not hasattr(args, 'config'): return args
        args = self.read_config(args, experiment)
        self.add_extras(args)
        self.set_seed(args)
        self.get_commit(args)
        self.generate_exp_name(args)
        self.mkdir(args)
        self.save_diff(args)
        args.task_type = "locomotion"
        args.obs_shape = [-1]

        if "MineRL" in args.dataset:
            args.task_type = "MineRL"
            args.obs_shape = [3, 64, 64]
        elif args.dataset in ["Breakout", "Pong", "Qbert", "Seaquest"]:
            args.task_type = "atari"
            args.obs_shape = [4, 84, 84]

        return args

    def read_config(self, args, experiment):
        '''
            Load parameters from config file
        '''
        dataset = args.dataset.replace('-', '_')
        print(f'[ utils/setup ] Reading config: {args.config}:{dataset}')
        module = importlib.import_module(args.config)
        params = getattr(module, 'base')[experiment]

        if hasattr(module, dataset) and experiment in getattr(module, dataset):
            print(f'[ utils/setup ] Using overrides | config: {args.config} | dataset: {dataset}')
            overrides = getattr(module, dataset)[experiment]
            params.update(overrides)
        else:
            print(f'[ utils/setup ] Not using overrides | config: {args.config} | dataset: {dataset}')

        for key, val in params.items():
            setattr(args, key, val)

        return args

    def add_extras(self, args):
        '''
            Override config parameters with command-line arguments
        '''
        extras = args.extra_args
        if not len(extras):
            return

        print(f'[ utils/setup ] Found extras: {extras}')
        assert len(extras) % 2 == 0, f'Found odd number ({len(extras)}) of extras: {extras}'
        for i in range(0, len(extras), 2):
            key = extras[i].replace('--', '')
            val = extras[i+1]
            assert hasattr(args, key), f'[ utils/setup ] {key} not found in config: {args.config}'
            old_val = getattr(args, key)
            old_type = type(old_val)
            print(f'[ utils/setup ] Overriding config | {key} : {old_val} --> {val}')
            if val == 'None':
                val = None
            elif val == 'latest':
                val = 'latest'
            elif old_type in [bool, type(None)]:
                val = eval(val)
            else:
                val = val
            setattr(args, key, val)

    def set_seed(self, args):
        if not 'seed' in dir(args):
            return
        set_seed(args.seed)

    def generate_exp_name(self, args):
        if not 'exp_name' in dir(args):
            return
        exp_name = getattr(args, 'exp_name')
        if callable(exp_name):
            exp_name_string = exp_name(args)
            print(f'[ utils/setup ] Setting exp_name to: {exp_name_string}')
            setattr(args, 'exp_name', exp_name_string)

    def mkdir(self, args):
        if 'logbase' in dir(args) and 'dataset' in dir(args) and 'exp_name' in dir(args):
            args.savepath = os.path.join(os.path.expanduser(args.logbase), args.dataset, args.exp_name)
            if 'suffix' in dir(args):
                args.savepath = os.path.join(os.path.expanduser(args.savepath), args.suffix)
            if mkdir(args.savepath):
                print(f'[ utils/setup ] Made savepath: {args.savepath}')
            self.save()

    def get_commit(self, args):
        args.commit = get_git_rev()

    def save_diff(self, args):
        try:
            save_git_diff(os.path.join(os.path.expanduser(args.savepath), 'diff.txt'))
        except:
            print('[ utils/setup ] WARNING: did not save git diff')
