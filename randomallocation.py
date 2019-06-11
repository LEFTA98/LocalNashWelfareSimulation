import numpy as np
import matplotlib.pyplot as plt

class Good:
    """Represents a single indivisible good."""
    
    def __init__(self, name, value):
        """
        Input: name (int), value (np.array of float)
        Output: initializes a new Good with the given name and value array,
        where the ith entry of the array represents agent i's value for this
        item.
        """
        self.name = name
        self.values = value
        
    def get_value(self, agent):
        """
        Input: agent (int)
        Output: the value the given agent has for the good.
        """
        return self.values[agent-1]
    
class Allocation:
    """Represents a binary-valued matrix modelling the allocation of what
    agents have what goods."""
    
    def __init__(self, mtx):
        """
        Input: binary-valued n x m matrix, where n is number of agents and m
        is number of goods. Each column must sum to 1.
        Output: Initializes a new Allocation object.
        """
        self.mtx = mtx
        
    def transfer(self, i, agent_1, agent_2):
        """
        Input: i (int), agent_1 (int), agent_2 (int)
        Output: Returns a new Allocation with the ith good of agent 1 given
        to agent 2 instead.
        """
        mtx = self.mtx.copy()
        mtx[agent_1-1][i-1] -= 1
        mtx[agent_2-1][i-1] += 1
        return Allocation(mtx)

    def get_all_local_allocations(self):
        """
        Input: None
        Output: Returns a new list of Allocations, old player, and new player, that contains all possible local changes from 
        the currently stored Allocation.
        """      

        matrix = self.mtx.copy()
        stored_matrices = []
        n,m = matrix.shape
        for i in range(n):
            for j in range(m):
                matrix = self.mtx.copy()
                vec = [0]*n
                vec[i] = 1
                old_player = matrix[:,j].argmax()
                matrix[:,j] = vec
                if not any([np.array_equal(item[0],matrix) for item in stored_matrices]):
                    stored_matrices.append([matrix, old_player, i])

        for item in stored_matrices:
            item[0] = Allocation(item[0])

        return stored_matrices
    
class Simulation:
    """
    Represents an instance of the indivisible goods problem. Contains a list 
    of the available Goods and an initial Allocation of those goods among the
    agents.
    """
    
    def __init__(self, goods, alloc):
        """
        Input: goods (1-d array of Good items), alloc (Allocation)
        Output: Initializes a new Simulation
        """
        self.goods = goods
        self.alloc = alloc
        self.good_mtx = np.array([good.values for good in self.goods], dtype=object)
        
    def get_nsw(self, alloc=None):
        """
        Input: alloc (Allocation)
        Output: Returns the Nash social welfare of the problem when the given
        alloc is used. Defaults to the stored Allocation if alloc is None.
        """
        if alloc is None:
            alloc = self.alloc
            
        temp = self.good_mtx * alloc.mtx.T
        temp = np.sum(temp,axis=0)
        temp = np.array([item for item in temp if item != 0], dtype=object)
        return np.product(temp)
    
    def get_agent_value(self, agent, alloc=None):
        """
        Input: alloc(Allocation), agent (int)
        Output: Returns the value of the given agent under the Allocation. Once
        again if the value of alloc is None the stored Allocation is used.
        """
        if alloc is None:
            alloc = self.alloc
            
            
        return np.sum(self.good_mtx[:,agent-1]*alloc.mtx[agent-1])
    
    def get_lower_bound_improvement(self):
        """
        Input: None
        Output: Returns the minimal possible improvement in NSW in each step
        as specified by the paper.
        """
        k = max(np.sum(self.good_mtx,axis=0))
        return 1 + 1/(k**2)

    def random_local_search(self, show_step=False, show_plot=False):
        """
        Input: None
        Output: an Allocation that achieves local Nash optimality, created by randomly switching goods of agents
        one at a time and the number of steps taken to get the allocation.
        """
        good_mtx = np.array([good.values for good in self.goods])
        unchanged = False
        ratios = []
        num_steps = 0

        if show_step:
            print("number of steps:", num_steps)
            temp = good_mtx * self.alloc.mtx.T
            print(temp)

        while not unchanged:
            allocation_candidates = self.alloc.get_all_local_allocations()
            np.random.shuffle(allocation_candidates)
            while len(allocation_candidates) > 0:
                allocation, start_player, end_player = allocation_candidates.pop()
                old_val = self.get_agent_value(start_player, None)*self.get_agent_value(end_player, None)
                new_val = self.get_agent_value(start_player, allocation)*self.get_agent_value(end_player, allocation)
                if (old_val < new_val):
                    self.alloc = allocation
                    ratios.append(new_val/old_val) if old_val != 0 else ratios.append(1) # 1 is appended if the old value had ratio 0
                    num_steps += 1
                    
                    if show_step:
                        print("number of steps:", num_steps)
                        temp = good_mtx * self.alloc.mtx.T
                        print(temp)

                    break
            
                if len(allocation_candidates) == 0:
                    unchanged = True

        if show_plot:
            x = np.arange(len(ratios))
            bound = self.get_lower_bound_improvement() * np.ones(len(ratios))
            plt.yscale('log')
            plt.plot(x, ratios, x, bound)
            plt.show()

        return self.alloc, num_steps
                
if __name__ == "__main__":
    NUM_PLAYERS = 2
    NUM_GOODS = 10
    final_allocs = []
    num_steps_list = []
    for i in range(5000):
        goods = []
        for j in range(NUM_GOODS):
            goods.append(Good(j, [np.random.randint(1,1001), np.random.randint(1,1001)]))
            
        mtx = []
        for i in range(1,NUM_PLAYERS+1):
            vec = np.zeros(NUM_GOODS)
            indices = np.arange(1,NUM_GOODS+1)
            vec = np.where((indices <= i/NUM_PLAYERS*NUM_GOODS) & ((i-1)/NUM_PLAYERS*NUM_GOODS < indices), 1, 0)
            mtx.append(vec)
            
        alloc = Allocation(np.array(mtx))
        sim = Simulation(goods, alloc)
        alloc, num_steps = sim.random_local_search(False, False)
        final_allocs.append(alloc)
        num_steps_list.append(num_steps)

    print(num_steps_list)
    print(max(num_steps_list))

    # goods = []
    # for j in range(NUM_GOODS):
    #     goods.append(Good(j, [np.random.randint(1,1001), np.random.randint(1,1001)]))

    # mtx = []
    # for i in range(1,NUM_PLAYERS+1):
    #     vec = np.zeros(NUM_GOODS)
    #     indices = np.arange(1,NUM_GOODS+1)
    #     vec = np.where((indices <= i/NUM_PLAYERS*NUM_GOODS) & ((i-1)/NUM_PLAYERS*NUM_GOODS < indices), 1, 0)
    #     mtx.append(vec)
            
    # alloc = Allocation(np.array(mtx))
    # sim = Simulation(goods, alloc)

    # good_mtx = np.array([good.values for good in sim.goods])
    # print(good_mtx)

    # alloc, num_steps = sim.random_local_search(True, True)

            