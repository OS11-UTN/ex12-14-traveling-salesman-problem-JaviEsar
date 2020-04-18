import numpy as np
import matplotlib.pyplot as plt

def plotTSP (graph, positions, result, nodes, time):
    graph.clear()
    graph.set_title('Current Best')
    graph.scatter(positions[:, 0], positions[:, 1])
    for i in range(nodes-1):
        graph.annotate("", xy=positions[result[i]], xycoords='data', xytext=positions[result[i+1]], textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    graph.annotate("", xy=positions[result[nodes-1]], xycoords='data', xytext=positions[result[0]], textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    if time == 0:
        plt.show()
    else:
        plt.draw()
        plt.pause(time)

# Control parameters
node_count = 30
population_size = 1000
elite_size = 200
mutation_rate = 0.001

# Cost to go from node 'row' to node 'column'
np.random.seed(133)
positions = np.random.rand(node_count, 2)
cnn = np.zeros([node_count, node_count])
for i in range(node_count-1):
    for j in range(i+1, node_count):
        cnn[i, j] = np.sqrt(pow(positions[i,0]-positions[j,0],2) + pow(positions[i,1]-positions[j,1],2))
        cnn[j, i] = cnn[i, j]

# Random population
population = np.zeros([population_size, node_count]).astype(int)
for i in range(population_size):
    notVisited = list(range(0, node_count))
    np.random.shuffle(notVisited)
    for j in range(node_count):
        population[i,j] = notVisited.pop()

stop = 0
cost = np.zeros(population_size)
best_cost = np.inf
best_tour = []
fig, graph = plt.subplots(1)
fig.canvas.set_window_title("TSP using Genetic Algorithms")
while stop != 10:
    # Tour ranking
    total_prob = 0
    prob = np.zeros(population_size)
    for i in range(population_size):
        if cost[i] == 0:
            for j in range(node_count - 1):
                cost[i] += cnn[population[i, j], population[i, j + 1]]
            cost[i] += cnn[population[i, node_count - 1], population[i, 0]]
        prob[i] = 1/cost[i]
        total_prob += prob[i]
    least_cost_idx = np.argsort(cost)
    prob /= total_prob

    # Compare with current best
    if cost[least_cost_idx[0]] < best_cost:
        best_cost = cost[least_cost_idx[0]]
        best_tour = population[least_cost_idx[0]]
        stop = 0
        plotTSP(graph, positions, best_tour, node_count, 0.001)
    else:
        stop += 1

    # Parent selection
    new_population = population[least_cost_idx[0:elite_size]]
    new_cost = cost[least_cost_idx[0:elite_size]]

    # Child creation
    for i in range(population_size-elite_size):
        parents = population[np.random.choice(least_cost_idx, size=2, p=prob)]
        gen1start = np.random.randint(0, node_count)
        gen1end = np.random.randint(gen1start+1, node_count+1)
        gen1 = parents[0,gen1start:gen1end]
        gen2 = [x for x in parents[1] if x not in gen1]
        child = np.concatenate((gen2[0:gen1start], gen1, gen2[gen1start:]), axis=0).astype(int)
        new_population = np.concatenate((new_population, [child]), axis=0)
        new_cost = np.concatenate((new_cost, [0]), axis=0)

    # Mutation
    for i in range(population_size):
        if np.random.rand() <= mutation_rate:
            pos1 = np.random.randint(0, node_count)
            pos2 = np.random.randint(0, node_count)
            new_population[i, pos1], new_population[i, pos2] = new_population[i, pos2], new_population[i, pos1]
            new_cost[i] = 0

    population = new_population
    cost = new_cost



# Print Result
print("Solver: Genetic Algorithm")
print("Best Tour: ", best_tour)
print("Cost: ", best_cost)
plotTSP(graph, positions, best_tour, node_count, 0)
