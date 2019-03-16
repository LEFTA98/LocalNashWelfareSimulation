import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

class Agent:
    def __init__(self, num_total_items, max_value_per_item, item_value=None):
        self.num_total_items = num_total_items
        self.max_value_per_item = max_value_per_item
        self.items_value = np.random.randint(low=1, high=max_value_per_item, size=num_total_items).astype(int)
        self.items = []

        # Normal distribution
        # scale = 3.
        # range = 10
        # size = 100000
        # X = truncnorm(a=1, b=max_value_per_item, scale=scale).rvs(size=num_total_items)
        # self.items_value = X.round().astype(int)

    def add_one_item(self, idx):
        idx = int(idx)
        if idx in self.items:
            print("Error: Agent already had the item!")
        elif type(idx) == int:
            self.items = np.append(self.items, idx)

    def delete_one_item(self, idx):
        idxidx = np.where(self.items == idx)
        self.items = np.delete(self.items, idxidx[0])

    def add_items(self, idx_lst):
        idx_lst = [int(x) for x in idx_lst]
        self.items = np.append(self.items, idx_lst)

    def get_max_value_per_item(self):
        return self.max_value_per_item

    def get_total_num_items(self):
        return self.num_total_items

    def get_items_value_table(self):
        return self.items_value

    def get_total_value(self):
        return sum([self.items_value[int(i)] for i in self.items])

    def get_items(self):
        return self.items

    def get_num_items(self):
        return len(self.items)

    def get_item_value(self, idx):
        return self.items_value[int(idx)]

    def __str__(self):
        return "Current items: " + " ".join(str(int(x)) for x in self.items) + \
               ", total value: {}".format(self.get_total_value())


class ExchangeSimulation:
    def __init__(self, num_items=10, max_value_per_item=80, num_agents=2):
        self.agent1 = Agent(num_total_items=num_items, max_value_per_item=max_value_per_item)
        self.agent2 = Agent(num_total_items=num_items, max_value_per_item=max_value_per_item)
        # self.agent1.items_value = np.arange(num_items) + 1
        # self.agent2.items_value = np.flip(self.agent1.items_value)
        # self.agent2.items_value = self.agent1.items_value
        self.history = []
        self.num_items = num_items
        self.max_value_per_item = max_value_per_item
        self.turn = 0
        self.random_allocation()

    def random_allocation(self):
        agent1_items = np.sort(np.random.choice(self.num_items, size=int(self.num_items/2), replace=False)).astype(int)
        agent2_items = np.setdiff1d(np.arange(self.num_items).astype(int), agent1_items)
        agent1_items = [int(x) for x in agent1_items]
        agent2_items = [int(x) for x in agent2_items]
        self.agent1.add_items(agent1_items)
        self.agent2.add_items(agent2_items)
        self.print_current_states()

    def print_current_states(self):
        print("================================================")
        #print("Random Initialization:")
        print("Agent 1:")
        print(self.agent1)
        print("Agent 2:")
        print(self.agent2)

    def one_exchange(self, agent, item):
        if agent == 1:
            self.agent1.delete_one_item(item)
            self.agent2.add_one_item(item)
        else:
            self.agent2.delete_one_item(item)
            self.agent1.add_one_item(item)
        agent1val = self.agent1.get_item_value(item)
        agent2val = self.agent2.get_item_value(item)
        if agent == 1:
            if agent1val > agent2val:
                print("Item is transferred to the agent with lower valuation.")
        else:
            if agent2val > agent1val:
                print("Item is transferred to the agent with lower valuation.")
        print("Item {}'s value: Agent 1 {} Agent 2 {}".format(item, self.agent1.get_item_value(item),
                                                              self.agent2.get_item_value(item)))
        self.print_current_states()

    def find_max_improvement(self):
        max_inc_item = None
        agent1_items = self.agent1.get_items()

        agent2_items = self.agent2.get_items()
        agent1_val = self.agent1.get_total_value()
        agent2_val = self.agent2.get_total_value()
        for item in agent1_items:
            item_val1 = self.agent1.get_item_value(item)
            item_val2 = self.agent2.get_item_value(item)
            nash_welfare_before = agent1_val * agent2_val
            nash_welfare_after = (agent1_val - item_val1) * (agent2_val + item_val2)
            if  nash_welfare_after > nash_welfare_before:
                if not max_inc_item:
                    max_inc_item = [1, item, nash_welfare_after, nash_welfare_before]
                elif max_inc_item[-2] < nash_welfare_after:
                    max_inc_item = [1, item, nash_welfare_after, nash_welfare_before]

        for item in agent2_items:
            item_val1 = self.agent1.get_item_value(item)
            item_val2 = self.agent2.get_item_value(item)
            nash_welfare_before = agent1_val * agent2_val
            nash_welfare_after = (agent1_val + item_val1) * (agent2_val - item_val2)
            if nash_welfare_after > nash_welfare_before:
                if not max_inc_item:
                    max_inc_item = [2, item, nash_welfare_after, nash_welfare_before]
                elif max_inc_item[-2] < nash_welfare_after:
                    max_inc_item = [2, item, nash_welfare_after, nash_welfare_before]
        return max_inc_item

    def max_exchange(self):
        max_ret = self.find_max_improvement()
        ratio = []
        item_lst = []
        while max_ret is not None:
            print("Agent {} gives item {} to Agent {} and the welfare is {} beofre exchange and is {} after exchange".
                  format(max_ret[0], int(max_ret[1]), int(3 - max_ret[0]), max_ret[3], max_ret[2]))
            self.one_exchange(max_ret[0], int(max_ret[1]))
            ratio.append(max_ret[-2]/max_ret[-1])
            item_lst.append(max_ret[0])
            max_ret = self.find_max_improvement()
        return ratio, item_lst

    def find_min_improvement(self):
        min_inc_item = None
        agent1_items = self.agent1.get_items()
        agent2_items = self.agent2.get_items()
        agent1_val = self.agent1.get_total_value()
        agent2_val = self.agent2.get_total_value()
        for item in agent1_items:
            item_val1 = self.agent1.get_item_value(item)
            item_val2 = self.agent2.get_item_value(item)
            nash_welfare_before = agent1_val * agent2_val
            nash_welfare_after = (agent1_val - item_val1) * (agent2_val + item_val2)
            if nash_welfare_after > nash_welfare_before:
                if not min_inc_item:
                    min_inc_item = [1, item, nash_welfare_after, nash_welfare_before]
                elif min_inc_item[-2] > nash_welfare_after:
                    min_inc_item = [1, item, nash_welfare_after, nash_welfare_before]

        for item in agent2_items:
            item_val1 = self.agent1.get_item_value(item)
            item_val2 = self.agent2.get_item_value(item)
            nash_welfare_before = agent1_val * agent2_val
            nash_welfare_after = (agent1_val + item_val1) * (agent2_val - item_val2)
            if nash_welfare_after > nash_welfare_before:
                if not min_inc_item:
                    min_inc_item = [2, item, nash_welfare_after, nash_welfare_before]
                elif min_inc_item[-2] > nash_welfare_after:
                    min_inc_item = [2, item, nash_welfare_after, nash_welfare_before]
        return min_inc_item

    def min_exchange(self):
        min_ret = self.find_min_improvement()
        ratio = []
        item_lst = []

        while min_ret is not None:
            print("Agent {} gives item {} to Agent {} and the welfare is {} beofre exchange and is {} after exchange".
                  format(min_ret[0], int(min_ret[1]), int(3 - min_ret[0]), min_ret[3], min_ret[2]))
            self.one_exchange(min_ret[0], int(min_ret[1]))
            ratio.append(min_ret[-2]/min_ret[-1])
            item_lst.append(min_ret[1])
            min_ret = self.find_min_improvement()
        return ratio, item_lst


game = ExchangeSimulation(400, max_value_per_item=90)
k = max(sum(game.agent1.get_items_value_table()), sum(game.agent2.get_items_value_table()))
ratio, item_lst = game.min_exchange()
factor = [1 + 1 / k ** 2] * len(ratio)
x = np.arange(len(ratio))
fig, ax = plt.subplots()
ax.plot(x, ratio, x, factor)
# plt.scatter(x, ratio, marker="x")

print("Num of goods {} {} after exchange.".format(game.agent1.get_num_items(), game.agent2.get_num_items()))
plt.show()
