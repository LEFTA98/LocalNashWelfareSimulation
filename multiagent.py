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
        self.good_mtx = np.array([good.values for good in self.goods])
        
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
        temp = [item for item in temp if item != 0]
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
    
    def max_NSW(self, show_step=False, show_plot=False):
        """
        Input: show_step (bool) whether or not to print a representation of 
        the allocations in each step, show_plot (bool) whether or not to print
        a representation of the NSW improvment ratio over time
        Output: Runs the local search algorithm to find an allocation that
        satisfies local Nash optimality, making the optimal move each step.
        """
        unchanged = False
        ratios = []
        num_steps = 0
        n = self.alloc.mtx.shape[0]
        m = self.alloc.mtx.shape[1]
        while not unchanged:
            num_steps += 1
            curr_change_candidate = self.alloc
            for i in range(1,m+1):
                for agent_1 in range(1, n+1):
                    for agent_2 in range(1, n+1):

                        if agent_1 != agent_2 and self.alloc.mtx[agent_1-1, i-1] == 1:
                            candidate = self.alloc.transfer(i, agent_1, agent_2)

                            old_prod = self.get_agent_value(agent_1,None)*self.get_agent_value(agent_2,None)
                            best_prod = self.get_agent_value(agent_1, curr_change_candidate)*self.get_agent_value(agent_2, curr_change_candidate)
                            new_prod = self.get_agent_value(agent_1, candidate)*self.get_agent_value(agent_2, candidate)
                            
                            if max(old_prod, best_prod) < new_prod:
                                curr_change_candidate = candidate
#                                
            if curr_change_candidate == self.alloc:
                unchanged = True
#           
            ratios.append(self.get_nsw(curr_change_candidate)/self.get_nsw())
            self.alloc = curr_change_candidate
                
            if show_step:
                print("number of steps:", num_steps)
                good_mtx = np.array([good.values for good in self.goods])
                temp = good_mtx * self.alloc.mtx.T
                print(temp)
            
        if show_plot:
            x = np.arange(len(ratios))
            bound = self.get_lower_bound_improvement() * np.ones(len(ratios))
            plt.plot(x, ratios, x, bound)
            
    def min_NSW(self, show_step=False, show_plot=False):
        """
        Input: show_step (bool) whether or not to print a representation of 
        the allocations in each step, show_plot (bool) whether or not to print
        a representation of the NSW improvment ratio over time
        Output: Runs the local search algorithm to find an allocation that
        satisfies local Nash optimality, making the least optimal move each
        step while still improving.
        """
        unchanged = False
        ratios = []
        num_steps = 0
        n = self.alloc.mtx.shape[0]
        m = self.alloc.mtx.shape[1]
        while not unchanged:
            num_steps += 1
            curr_change_candidate = self.alloc
            to_add = 1
            for i in range(1,m+1):
                for agent_1 in range(1, n+1):
                    for agent_2 in range(1, n+1):

                        if agent_1 != agent_2 and self.alloc.mtx[agent_1-1, i-1] == 1:
                            candidate = self.alloc.transfer(i, agent_1, agent_2)

                            old_prod = self.get_agent_value(agent_1,None)*self.get_agent_value(agent_2,None)
                            best_prod = self.get_agent_value(agent_1, curr_change_candidate)*self.get_agent_value(agent_2, curr_change_candidate)
                            new_prod = self.get_agent_value(agent_1, candidate)*self.get_agent_value(agent_2, candidate)
                            
                            if old_prod < new_prod and ((new_prod < best_prod) or (best_prod == old_prod)):
                                curr_change_candidate = candidate
                                to_add = np.exp(np.log(new_prod)-np.log(old_prod))
#                                
            if curr_change_candidate == self.alloc:
                unchanged = True
                
            self.alloc = curr_change_candidate
            
            if show_plot:
                if self.get_nsw() != 0:
                    ratios.append(to_add)
                else:
                    #Numerical stability if NSW is not 0
                    ratios.append(self.get_nsw(curr_change_candidate)/self.get_nsw())
                
            if show_step:
                print("number of steps:", num_steps)
                good_mtx = np.array([good.values for good in self.goods])
                temp = good_mtx * self.alloc.mtx.T
                print(temp)
          
        if show_plot:
            x = np.arange(len(ratios))
            bound = self.get_lower_bound_improvement() * np.ones(len(ratios))
            plt.plot(x, ratios, x, bound)
            
                
if __name__ == "__main__":
    #Toy example
#    good1 = Good(1, np.array([1,1,1]))
#    good2 = Good(2, np.array([2,2,2]))
#    good3 = Good(3, np.array([3,3,3]))
#    good4 = Good(4, np.array([4,4,4]))
#    good5 = Good(5, np.array([5,5,5]))
#    
#    goods = [good1, good2, good3, good4, good5]
#    mtx = np.array([[1,1,1,1,1],[0,0,0,0,0],[0,0,0,0,0]])
#    alloc = Allocation(mtx)
#    
#    sim = Simulation(goods, alloc)
#    sim.min_NSW(True, True)
    
    #Discrete Uniform Random example
    NUM_PLAYERS = 6
    NUM_GOODS = 200
    MAX_VALUE = 15
    goods = []
    for j in range(NUM_GOODS):
        goods.append(Good(j, np.random.randint(1, MAX_VALUE, size=NUM_PLAYERS)))
        
    mtx = []
    for i in range(1,NUM_PLAYERS+1):
        vec = np.zeros(NUM_GOODS)
        indices = np.arange(1,NUM_GOODS+1)
        vec = np.where((indices <= i/NUM_PLAYERS*NUM_GOODS) & ((i-1)/NUM_PLAYERS*NUM_GOODS < indices), 1, 0)
        mtx.append(vec)
        
    alloc = Allocation(np.array(mtx))
    
    sim = Simulation(goods, alloc)
    sim.min_NSW(False, True)

            