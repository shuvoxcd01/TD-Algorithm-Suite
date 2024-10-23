from abc import abstractmethod

from mc_suite.policies.base_policy import BasePolicy


class BaseSoftPolicy(BasePolicy):

    @abstractmethod
    def get_action_deterministic(self, state):
        raise NotImplementedError("This method must be overridden")
