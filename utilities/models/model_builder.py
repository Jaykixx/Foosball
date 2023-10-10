from rl_games.algos_torch import network_builder
from rl_games.common import object_factory
from rl_games.algos_torch import models
import rl_games.algos_torch

from utilities.models.ndp import NDP, NDPA2CBuilder


NETWORK_REGISTRY = {}
MODEL_REGISTRY = {}


def register_network(name, target_class):
    NETWORK_REGISTRY[name] = lambda **kwargs: target_class()


def register_model(name, target_class):
    MODEL_REGISTRY[name] = lambda  network, **kwargs: target_class(network)


class CustomNetworkBuilder:
    def __init__(self):
        self.network_factory = object_factory.ObjectFactory()
        self.network_factory.set_builders(NETWORK_REGISTRY)
        self.network_factory.register_builder(
            'actor_critic',
            lambda **kwargs: network_builder.A2CBuilder()
        )
        self.network_factory.register_builder(
            'resnet_actor_critic',
            lambda **kwargs: network_builder.A2CResnetBuilder()
        )
        self.network_factory.register_builder(
            'rnd_curiosity',
            lambda **kwargs: network_builder.RNDCuriosityBuilder()
        )
        self.network_factory.register_builder(
            'soft_actor_critic',
            lambda **kwargs: network_builder.SACBuilder()
        )
        self.network_factory.register_builder(
            'ndp_actor_critic',
            lambda **kwargs: NDPA2CBuilder()
        )

    def load(self, params):
        model_name = params['model']['name']
        network_name = params['network']['name']
        if model_name == 'ndp':
            if network_name == 'actor_critic':
                network_name = model_name + "_" + network_name
            else:
                raise NotImplementedError
        network = self.network_factory.create(network_name)
        network.load(params['network'])

        return network


class CustomModelBuilder:
    def __init__(self):
        self.model_factory = object_factory.ObjectFactory()
        self.model_factory.set_builders(MODEL_REGISTRY)
        self.model_factory.register_builder(
            'discrete_a2c',
            lambda network, **kwargs: models.ModelA2C(network)
        )
        self.model_factory.register_builder(
            'multi_discrete_a2c',
            lambda network, **kwargs: models.ModelA2CMultiDiscrete(network)
        )
        self.model_factory.register_builder(
            'continuous_a2c',
            lambda network, **kwargs: models.ModelA2CContinuous(network)
        )
        self.model_factory.register_builder(
            'continuous_a2c_logstd',
            lambda network, **kwargs: models.ModelA2CContinuousLogStd(network)
        )
        self.model_factory.register_builder(
            'soft_actor_critic',
            lambda network, **kwargs: models.ModelSACContinuous(network)
        )
        self.model_factory.register_builder(
            'central_value',
            lambda network, **kwargs: models.ModelCentralValue(network)
        )
        self.model_factory.register_builder(
            'ndp',
            lambda network, **kwargs: NDP(network)
        )

        self.network_builder = CustomNetworkBuilder()

    def get_network_builder(self):
        return self.network_builder

    def load(self, params):
        model_name = params['model']['name']
        network = self.network_builder.load(params)
        model = self.model_factory.create(model_name, network=network)
        return model