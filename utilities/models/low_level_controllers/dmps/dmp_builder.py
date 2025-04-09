from rl_games.common import object_factory
from utilities.models.low_level_controllers.dmps.discrete import DiscreteDMP
from utilities.models.low_level_controllers.dmps.rythmic import RythmicDMP
from utilities.models.low_level_controllers.dmps.striking import StrikingDMP


MODEL_REGISTRY = {}


def register_model(name, target_class):
    MODEL_REGISTRY[name] = lambda network, **kwargs: target_class(network)


class DMPBuilder:
    def __init__(self):
        self.dmp_factory = object_factory.ObjectFactory()
        self.dmp_factory.set_builders(MODEL_REGISTRY)
        self.dmp_factory.register_builder(
            'discrete',
            lambda config, **kwargs: DiscreteDMP(config, **kwargs)
        )
        self.dmp_factory.register_builder(
            'rythmic',
            lambda config, **kwargs: RythmicDMP(config, **kwargs)
        )
        self.dmp_factory.register_builder(
            'striking',
            lambda config, **kwargs: StrikingDMP(config, **kwargs)
        )

    def build(self, config, **kwargs):
        dmp_type = config['type']
        return self.dmp_factory.create(dmp_type, config=config, **kwargs)
