#!/usr/bin/env python3
from gym.envs.registration import register, make, registry, spec
from gym.envs.registration import load_env_plugins as _load_env_plugins

register(
    id='MUMT_v5-v0',
    entry_point='gymPX4.envs.MUMT_v5:MUMT_v5'
)