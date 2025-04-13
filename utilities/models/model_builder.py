from rl_games.algos_torch.model_builder import *
from utilities.models.networks.oc_transformer import OCTBuilder


class ExtendedNetworkBuilder(NetworkBuilder):

    def __init__(self):
        NetworkBuilder.__init__(self)

        # Add new networks here
        self.register_network('oc_transformer', OCTBuilder)

    def load(self, params):
        network_name = params['network']['name']
        network = self.network_factory.create(network_name)
        network.load(params['network'])

        return network

    def register_network(self, name: str, constructor: callable):
        self.network_factory.register_builder(name, lambda **kwargs: constructor())


class CustomModelBuilder:
    def __init__(self):
        self.model_factory = object_factory.ObjectFactory()
        self.model_factory.set_builders(MODEL_REGISTRY)

        self.register_model('discrete_a2c', models.ModelA2C)
        self.register_model('multi_discrete_a2c', models.ModelA2CMultiDiscrete)
        self.register_model('continuous_a2c', models.ModelA2CContinuous)
        self.register_model('continuous_a2c_logstd', models.ModelA2CContinuousLogStd)
        self.register_model('soft_actor_critic', models.ModelSACContinuous)
        self.register_model('central_value', models.ModelCentralValue)

        # Add new models here

        self.network_builder = ExtendedNetworkBuilder()

    def get_network_builder(self):
        return self.network_builder

    def load(self, params):
        model_name = params['model']['name']
        network = self.network_builder.load(params)
        model = self.model_factory.create(model_name, network=network)
        return model

    def register_model(self,name: str, constructor: callable):
        self.model_factory.register_builder(
            name, lambda network, **kwargs: constructor(network)
        )
