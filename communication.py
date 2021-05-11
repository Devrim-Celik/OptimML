# Class to manage communication between agents
# If we want to try sharing gradients can do here
# Add differential privacy here or compression
import abc


class Communication(abc.ABC):
    def __init__(self, agent_list):
        self.agent_list = agent_list

    @abc.abstractmethod
    def send_updates(self):
        pass


class CentralizedCommunication(Communication):
    pass


class DecentralizedCommunication(Communication):
    def __init__(self, agent_list):
        super().__init__(agent_list)

    def _add_differential_privacy(self):
        pass

    def send_updates(self):
        for send_indx, sender_agent in enumerate(self.agent_list):
            for rec_indx, receiver_agent in enumerate(self.agent_list):
                if rec_indx in sender_agent.neighbours:
                    receiver_agent.receive_weights(sender_agent._weights())
