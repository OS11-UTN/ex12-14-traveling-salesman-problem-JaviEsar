import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# Converts a Cost Node-Node matrix to the corresponding SingleEntry-SingleExit and Cost matrix
def cnn2sesec (cnn):
    # Find the existing arcs
    arcs = np.argwhere(cnn)

    # Create the node-arc matrix and cost matrix
    na = np.zeros([2*cnn.shape[0], arcs.shape[0]]).astype(int)
    c = np.zeros([arcs.shape[0]])

    # For each arc, update the two corresponding entries in the node-arc matrix
    for i in range(arcs.shape[0]):
        na[arcs[i, 0], i] = 1
        na[cnn.shape[0]+arcs[i, 1], i] = 1
        c[i] = cnn[arcs[i, 0], arcs[i, 1]]

    # Return
    return na, c, arcs


def plotTSP (positions, arcs, result, nodes):

    fig, p1 = plt.subplots(1)  # Prepare 2 plots
    p1.set_title('Solution with Subtours')
    p1.scatter(positions[:, 0], positions[:, 1])
    for i in range(nodes):
        currArc = np.argmax(result)
        startNode = int(arcs[currArc, 0])
        endNode = int(arcs[currArc, 1])
        p1.annotate("", xy=positions[startNode], xycoords='data', xytext=positions[endNode], textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        result[currArc] = 0

    plt.show()


# Cost to go from node 'row' to node 'column'
np.random.seed(133)
positions = np.random.rand(6, 2)
cnn = np.zeros([positions.shape[0], positions.shape[0]])
for i in range(positions.shape[0]-1):
    for j in range(i+1, positions.shape[0]):
        cnn[i, j] = np.sqrt(pow(positions[i,0]-positions[j,0],2) + pow(positions[i,1]-positions[j,1],2))
        cnn[j, i] = cnn[i, j]

# Node-Arc Matrix and Cost Matrix computation
A, C, arcs = cnn2sesec(cnn)

# Net flow for each node (>0 is a source / <0 is a sink)
B = np.ones(A.shape[0])

# Decision variables bounds
bounds = tuple([0, None] for arcs in range(C.shape[0]))




# Solve the linear program using simplex
res = linprog(C, A_eq=A, b_eq=B, bounds=bounds, method='revised simplex')


# Print Result
print("Solver: Simplex")
print("Raw Solution: ", res.x)
print("Transported Units:")
for i in range(res.x.shape[0]):
    if res.x[i]!=0:
        print(arcs[i]+1, " -> ", res.x[i])
print("Objective Function Value: ", res.fun)

plotTSP(positions, arcs, res.x, cnn.shape[0])

print("The restrictions forced the solution to only have one incoming and outgoing arc for each node, however they do\n"
      "not assure that all the nodes will be joined in a single path, allowing the formation of smaller circuits or\n"
      "subtours. These can be removed by adding restrictions that impose that at least one of the nodes in one\n"
      "subtour must be connected to at least one of the nodes in the other subtour.")
