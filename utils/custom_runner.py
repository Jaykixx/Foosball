from rl_games.torch_runner import Runner

import utils.models.agents as agents
import utils.models.players as players


class CustomRunner(Runner):
    def __init__(self, algo_observer=None):
        super().__init__(algo_observer)
        self.algo_factory.register_builder(
            'a2c_continuous', lambda **kwargs: agents.A2CAgent(**kwargs)
        )
        self.player_factory.register_builder(
            'a2c_continuous', lambda **kwargs: players.A2CPlayer(**kwargs)
        )
