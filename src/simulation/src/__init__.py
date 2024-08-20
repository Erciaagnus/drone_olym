#!/usr/bin/env python3
from gym.envs.registration import register, make, registry, spec
from gym.envs.registration import load_env_plugins as _load_env_plugins

register(
    id='SUST_v3-v0',
    entry_point='simulation.src.envs.sust_v3:SUST_v3',
)
register(
    id='DKC_real_Unicycle',
    entry_point='simulation.src.envs.dkc_unicycle_realUAV:DKC_real_Unicycle',
)


register(
    id='SUST_v4-v0',
    entry_point='simulation.src.envs.sust_v4:SUST_v4',
)