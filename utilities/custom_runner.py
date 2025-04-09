from rl_games.torch_runner import Runner, _restore, _override_sigma
from rl_games.common import object_factory

import utilities.models.agents as agents
import utilities.models.players as players


class CustomRunner(Runner):
    def __init__(self, algo_observer=None):
        super().__init__(algo_observer)

        # Customized Default
        self.algo_factory.register_builder(
            'a2c_continuous', lambda **kwargs: agents.A2CAgent(**kwargs)
        )
        self.player_factory.register_builder(
            'a2c_continuous', lambda **kwargs: players.A2CPlayer(**kwargs)
        )

        # ASE implementation
        self.algo_factory.register_builder(
            'ase_a2c_continuous', lambda **kwargs: agents.ase_agent(**kwargs)
        )
        self.player_factory.register_builder(
            'ase_a2c_continuous', lambda **kwargs: players.ase_player(**kwargs)
        )

        # Add wrapper factory to implement extensions to A2C-based algorithms
        # e.g. A2C-Base and ASE
        self.wrapper_factory = object_factory.ObjectFactory()
        # WocaR implementation
        self.wrapper_factory.register_builder(
            'wocar', lambda **kwargs: agents.WocarWrapper(**kwargs)
        )

    def load_config(self, params):
        Runner.load_config(self, params)
        self.algo_wrapper = self.algo_params.get('wrapper', None)

    def run_train(self, args):
        print('Started to train')
        agent = self.algo_factory.create(self.algo_name, base_name='run', params=self.params)
        if self.algo_wrapper is not None and self.algo_wrapper != '':
            agent = self.wrapper_factory.create(self.algo_wrapper, base_agent=agent, params=self.params)
        _restore(agent, args)
        _override_sigma(agent, args)
        agent.train()

    def run_evaluation(self, args):
        if 'checkpoint' in args and args['checkpoint'] is not None and args['checkpoint'] !='':
            checkpoint = args['checkpoint']
            print(f'Starting Evaluation for Checkpoint: {checkpoint}')
        else:
            print('Starting Evaluation without Checkpoint')
        player = self.create_player()
        _restore(player, args)
        _override_sigma(player, args)
        return player.evaluate()

    def run_physical_system(self, args):
        if 'checkpoint' in args and args['checkpoint'] is not None and args['checkpoint'] !='':
            checkpoint = args['checkpoint']
            print(f'Starting Evaluation for Checkpoint: {checkpoint}')
        else:
            print('Starting Evaluation without Checkpoint')
        player = self.create_player()
        _restore(player, args)
        _override_sigma(player, args)
        return player.run_physical_env()
