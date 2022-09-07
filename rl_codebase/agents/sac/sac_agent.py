from rl_codebase.cmn import (
	ReplayBuffer,
	evaluate_policy,
	collect_transitions,
	Logger,
)
from .sac_continuous import ContinuousSAC

class SAC:
	def __init__(
		self, 
		env,
		learning_rate:float=3e-4,
		buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
	):
		self.agent = ContinuousSAC(env, learning_rate)