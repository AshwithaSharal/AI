# AI
# Write a Program to Implement Breadth First Search using Python.
graph={<br>
    '1':['2','10'],'1':['2','10'],<br>
    '2':['3','8'],<br>
    '3':['4'],<br>
    '4':['5','6','7'],<br>
    '5':[],<br>
    '6':[],<br>
    '7':[],<br>
    '8':['9'],<br>
    '9':[],<br>
    '10':[]<br>
}<br>
visited=[]<br>
queue=[]<br>
def bfs(visited,graph,node):<br>
    visited.append(node)<br>
    queue.append(node)<br>
    while queue:<br>
        m=queue.pop(0)<br>
        print(m,end=" ")<br>
        for neighbour in graph[m]:<br>
         if neighbour not in visited:<br>
            visited.append(neighbour)<br>
            queue.append(neighbour)<br>
print("Following is the Breadth-First Search")<br>
bfs(visited,graph,'1')<br>
*OUTPUT:*<br>
![image](https://user-images.githubusercontent.com/98145023/209536712-bd97aaaa-090a-4b12-a214-3d76ad8ed644.png)<br><br><br>

# Write a Program to Implement Depth First Search using Python.
graph = {<br>
'5': ['3','7'],<br>
'3': ['2','4'],<br>
'7': ['6'],<br>
'6':[],<br>
'2': ['1'],<br>
'1':[],<br>
'4': ['8'],<br>
'8':[]<br>
}<br>
visited = set() # Set to keep track of visited nodes of graph.<br>
def dfs(visited, graph, node): #function for dfs<br>
 if node not in visited:<br>
  print (node)<br>
  visited.add(node)<br>
  for neighbour in graph[node]:<br>
   dfs(visited, graph, neighbour)<br>
** Driver Code**<br>
print("Following is the Depth-First Search")<br>
dfs(visited, graph, '5')<br>
*OUTPUT:*<br>
![image](https://user-images.githubusercontent.com/98145023/209542631-f41995f9-d273-4e06-9ca0-6d585c0c8a44.png)<br><br><br>

# Write a Program to Implement Best First Search using Python.
from queue import PriorityQueue<br>
import matplotlib.pyplot as plt<br>
import networkx as nx<br>
**for implementing BFS | returns path having lowest cost**<br>
def best_first_search(source, target, n):<br>
    visited = [0] * n<br>
    visited[source] = True<br>
    pq = PriorityQueue()<br>
    pq.put((0, source))<br>
    while pq.empty() == False:<br>
        u = pq.get()[1]<br>
        print(u, end=" ") # the path having lowest cost<br>
        if u == target:<br>
            break<br>

        for v, c in graph[u]:<br>
            if visited[v] == False:<br>
                visited[v] = True<br>
                pq.put((c, v))<br>
    print()<br>
** for adding edges to graph**<br>
def addedge(x, y, cost):<br>
    graph[x].append((y, cost))<br>
    graph[y].append((x, cost))<br>

v = int(input("Enter the number of nodes: "))<br>
graph = [[] for i in range(v)] # undirected Graph<br>
e = int(input("Enter the number of edges: "))<br>
print("Enter the edges along with their weights:")<br>
for i in range(e):<br>
    x, y, z = list(map(int, input().split()))<br>
    addedge(x, y, z)<br>
source = int(input("Enter the Source Node: "))<br>
target = int(input("Enter the Target/Destination Node: "))<br>
print("\nPath: ", end = "")<br>
best_first_search(source, target, v)<br>
*OUTPUT:*<br>
![image](https://user-images.githubusercontent.com/98145023/209542744-8aeb63e0-c6cf-479e-9af8-20101ce4b09b.png)<br><br><br>



# Write a Program to Implement Water-Jug Problem using Python.from collections import defaultdict<br>
jug1, jug2, aim = 4, 3, 2<br>
visited = defaultdict(lambda: False)<br>
def waterJugSolver(amt1, amt2):<br>
    if (amt1 == aim and amt2 == 0) or (amt2 == aim and amt1 == 0):<br>
        print(amt1, amt2)<br>
        return True<br>
    if visited[(amt1, amt2)] == False:<br>
        print(amt1, amt2)<br>
        visited[(amt1, amt2)] = True<br>
        return (waterJugSolver(0, amt2) or<br>
              waterJugSolver(amt1, 0) or<br>
              waterJugSolver(jug1, amt2) or<br>
              waterJugSolver(amt1, jug2) or<br>
              waterJugSolver(amt1 + min(amt2, (jug1-amt1)),<br>
              amt2 - min(amt2, (jug1-amt1))) or<br>
              waterJugSolver(amt1 - min(amt1, (jug2-amt2)),<br>
              amt2 + min(amt1, (jug2-amt2))))<br>
    else:<br>
        return False<br>
print("Steps: ")<br>
waterJugSolver(0, 0)<br>
*OUTPUT:*<br>
![image](https://user-images.githubusercontent.com/98145023/209542842-8f269020-5291-4a66-b1e2-206b0926bccc.png)<br><br><br>


# Write a Program to Implement Tower of Hanoi usingPython.
def TowerOfHanoi(n , source, destination, auxiliary):<br>
    if n==1:<br>
        print ("Move disk 1 from source",source,"to destination",destination)<br>
        return<br>
    TowerOfHanoi(n-1, source, auxiliary, destination)<br>
    print ("Move disk",n,"from source",source,"to destination",destination)<br>
    TowerOfHanoi(n-1, auxiliary, destination, source)<br>
n = 3<br>
TowerOfHanoi(n,'A','B','C')<br>
*OUTPUT:*<br>
![image](https://user-images.githubusercontent.com/98145023/209542939-fa0a67f8-7462-46d9-934c-da86e575b4c1.png)<br><br><br>

# Write a Program to Implement Tic-Tac-Toe application using Python.
import numpy as np<br>
import random<br>
from time import sleep<br>
def create_board():<br>
    return(np.array([[0, 0, 0],<br>
    [0, 0, 0],<br>
    [0, 0, 0]]))<br>
def possibilities(board):<br>
    l = []<br>

    for i in range(len(board)):<br>
        for j in range(len(board)):<br>
          if board[i][j] == 0:<br>
                l.append((i, j))<br>
    return(l)<br>
def random_place(board, player):<br>
    selection = possibilities(board)<br>
    current_loc = random.choice(selection)<br>
    board[current_loc] = player<br>
    return(board)<br>
def row_win(board, player):<br>
    for x in range(len(board)):<br>
        win = True<br>

        for y in range(len(board)):<br>
            if board[x, y] != player:<br>
                win = False<br>
                continue<br>

        if win == True:<br>
            return(win)<br>
    return(win)<br>
def col_win(board, player):<br>
    for x in range(len(board)):<br>
        win = True<br>

        for y in range(len(board)):<br>
            if board[y][x] != player:<br>
                win = False<br>
                continue<br>

        if win == True:<br>
            return(win)<br>
    return(win)<br>
def diag_win(board, player):<br>
    win = True<br>
    y = 0<br>
    for x in range(len(board)):<br>
        if board[x, x] != player:<br>
            win = False<br>
    if win:<br>
        return win<br>
    win = True<br>
    if win:<br>
        for x in range(len(board)):<br>
            y = len(board) - 1 - x<br>
            if board[x, y] != player:<br>
                win = False<br>
    return win<br>
def evaluate(board):<br>
    winner = 0<br>

    for player in [1, 2]:<br>
        if (row_win(board, player) or<br>
            col_win(board,player) or<br>
            diag_win(board,player)):<br>
            winner = player<br>
    if np.all(board != 0) and winner == 0:<br>
        winner = -1<br>
    return winner<br>
def play_game():<br>
    board, winner, counter = create_board(), 0, 1<br>
    print(board)<br>
    sleep(2)<br>

    while winner == 0:<br>
        for player in [1, 2]:<br>
            board = random_place(board, player)<br>
            print("Board after " + str(counter) + " move")<br>
            print(board)<br>
            sleep(2)<br>
            counter += 1<br>
            winner = evaluate(board)<br>
            if winner != 0:<br>
                break<br>
    return(winner)<br>
print("Winner is: " + str(play_game()))<br>
*OUTPUT:*<br>
![image](https://user-images.githubusercontent.com/98145023/209543038-6473112b-f656-46f6-85e4-80081227166e.png)<br><br><br>


# Write a Program to Implement 8-Puzzle Problem using Python.
**Importing copy for deepcopy function**<br>
import copy<br>
**Importing the heap functions from python**<br>
**library for Priority Queue**<br>
from heapq import heappush, heappop<br>
**This variable can be changed to change**<br>
**the program from 8 puzzle(n=3) to 15**<br>
**puzzle(n=4) to 24 puzzle(n=5)...**<br>
n = 3<br>
**bottom, left, top, right** <br>
row = [ 1, 0, -1, 0 ]<br>
col = [ 0, -1, 0, 1 ]<br>
**A class for Priority Queue**<br>
class priorityQueue:<br>
**Constructor to initialize a Priority Queue<br><br>
	def __init__(self):<br><br>
		self.heap = []<br><br>

** Inserts a new key 'k'**<br>
	def push(self, k):<br>
		heappush(self.heap, k)<br>

**Method to remove minimum element from Priority Queue**<br>
	def pop(self):<br>
		return heappop(self.heap)<br>
**Method to know if the Queue is empty**<br>
	def empty(self):<br>
		if not self.heap:<br>
			return True<br>
		else:<br>
			return False<br>
** Node structure **<br>
class node:<br>
	
	def __init__(self, parent, mat, empty_tile_pos,<br>
				cost, level):<br>
**Stores the parent node of the current node helps in tracing path when the answer is found**<br>
		self.parent = parent<br>
** Stores the matrix**<br>
		self.mat = mat<br>
		**Stores the position at which the empty space tile exists in the matrix**<br>
		self.empty_tile_pos = empty_tile_pos<br>
**Storesthe number of misplaced tiles**<br>
		self.cost = cost<br>
**Stores the number of moves so far**<br>
		self.level = level<br>
**This method is defined so that the priority queue is formed based on the cost variable of the objects**<br>
	def __lt__(self, nxt):<br>
		return self.cost < nxt.cost<br>
** Function to calculate the number of misplaced tiles ie. number of non-blank  tiles not in their goal position**<br>
def calculateCost(mat, final) -> int:<br>
	
	count = 0<br>
	for i in range(n):<br>
		for j in range(n):<br>
			if ((mat[i][j]) and<br>
				(mat[i][j] != final[i][j])):<br>
				count += 1<br>
				
	return count<br>

def newNode(mat, empty_tile_pos, new_empty_tile_pos,<br>
			level, parent, final) -> node:<br>
				
**Copy data from parent matrix to current matrix**<br>
	new_mat = copy.deepcopy(mat)<br>
** Move tile by 1 position**<br>
	x1 = empty_tile_pos[0]<br>
	y1 = empty_tile_pos[1]<br>
	x2 = new_empty_tile_pos[0]<br>
	y2 = new_empty_tile_pos[1]<br>
	new_mat[x1][y1], new_mat[x2][y2] = new_mat[x2][y2], new_mat[x1][y1]<br>
**Set number of misplaced tiles**<br>
	cost = calculateCost(new_mat, final)<br>

	new_node = node(parent, new_mat, new_empty_tile_pos,<br>
					cost, level)<br>
	return new_node<br>
** Function to print the N x N matrix**<br>
def printMatrix(mat):<br>
	
	for i in range(n):<br>
		for j in range(n):<br>
			print("%d " % (mat[i][j]), end = " ")<br>
			
		print()
**Function to check if (x, y) is a valid matrix coordinate**<br>
def isSafe(x, y):<br>
	
	return x >= 0 and x < n and y >= 0 and y < n<br>
** Print path from root node to destination node<br>
def printPath(root):<br>
	
	if root == None:<br>
		return<br>
	
	printPath(root.parent)<br>
	printMatrix(root.mat)<br>
	print()<br>

**Function to solve N*N - 1 puzzle algorithm using Branch and Bound. empty_tile_pos is the blank tile position in the initial state.**<br>
def solve(initial, empty_tile_pos, final):<br>
** Create a priority queue to store live  nodes of search tree<br>
	pq = priorityQueue()<br>
** Create the root node**<br>
	cost = calculateCost(initial, final)<br>
	root = node(None, initial,<br>
				empty_tile_pos, cost, 0)<br>
** Add root to list of live nodes**<br>
	pq.push(root)<br>
** Finds a live node with least cost, add its children to list of live nodes and finally deletes it from the list.**<br>
	while not pq.empty():<br>

**Find a live node with least estimated cost and delete it form the list of live nodes**<br>
		minimum = pq.pop()<br>
**If minimum is the answer node**<br>
		if minimum.cost == 0:<br>
**Print the path from root to destination;**<br>
			printPath(minimum)<br>
			return<br>
**Generate all possible children**<br>
		for i in range(n):<br>
			new_tile_pos = [<br>
				minimum.empty_tile_pos[0] + row[i],<br>
				minimum.empty_tile_pos[1] + col[i], ]<br>
				
			if isSafe(new_tile_pos[0], new_tile_pos[1]):<br>
**Create a child node**<br>
				child = newNode(minimum.mat,<br>
								minimum.empty_tile_pos,<br>
								new_tile_pos,<br>
								minimum.level + 1,<br>
								minimum, final,)<br>

**Add child to list of live nodes**<br>
				pq.push(child)<br>

** Driver Code**<br>
**Initial configuration**<br>
**Value 0 is used for empty space**<br>
initial = [ [ 1, 2, 3 ],<br>
			[ 5, 6, 0 ],<br>
			[ 7, 8, 4 ] ]<br>

** Solvable Final configuration**<br>
** Value 0 is used for empty space**<br>
final = [ [ 1, 2, 3 ],<br>
		[ 5, 8, 6 ],<br>
		[ 0, 7, 4 ] ]<br>
** Blank tile coordinates in initial configuration**<br>
empty_tile_pos = [ 1, 2 ]<br>
**Function call to solve the puzzle**<br>
solve(initial, empty_tile_pos, final)<br>
*OUTPUT:*<br>
![image](https://user-images.githubusercontent.com/98145023/209543150-703d1782-a378-4167-9ba0-d331c719e337.png)<br><br><br>



# Write a Program to Implement Travelling Salesman problem using Python.<br>
from sys import maxsize<br>
from itertools import permutations<br>
V = 4<br>

def travellingSalesmanProblem(graph, s):<br>
**store all vertex apart from source vertex**<br>
  vertex = []<br>
  for i in range(V):<br>
   if i != s:<br>
    vertex.append(i)<br>
** store minimum weight Hamiltonian Cycle**<br>
    min_path = maxsize<br>
    next_permutation=permutations(vertex)<br>
  for i in next_permutation:<br>
** store current Path weight(cost)**<br>
        current_pathweight = 0<br>
**compute current path weight**<br>
        k = s<br>
        for j in i:<br>
         current_pathweight += graph[k][j]<br>
         k = j<br>
        current_pathweight += graph[k][s]<br>
** Update minimum**<br>
        min_path = min(min_path, current_pathweight)<br>
  return min_path<br>
** Driver Code**<br>
if __name__ == "__main__":<br>
**matrix representation of graph**<br>
 graph = [[0, 10, 15, 20], [10, 0, 35, 25],<br>
         [15, 35, 0, 30], [20, 25, 30, 0]]<br>
s = 0<br>
print(travellingSalesmanProblem(graph, s))<br>
*OUTPUT:*<br>
![image](https://user-images.githubusercontent.com/98145023/209544343-85347d2c-3c18-4b4a-bc62-2e75a0ea5b5d.png)<br><br><br>


# Write a program to implement the FIND-S Algorithm for finding the most specific hypothesis based on a given set of training data samples. Read the training data from a
.CSV file.
import pandas as pd<br>
import numpy as np<br>
 
**to read the data in the csv file**<br>
data = pd.read_csv("Train.csv")<br>
print(data)<br>
 
**making an array of all the attributes**<br>
d = np.array(data)[:,:-1]<br>
print("The attributes are: ",d)<br>
 
**segragating the target that has positive and negative examples**<br>
target = np.array(data)[:,-1]<br>
print("The target is: ",target)<br>
 
**training function to implement find-s algorithm**<br>
def train(c,t):<br>
    for i, val in enumerate(t):<br>
        if val == "Yes":<br>
            specific_hypothesis = c[i].copy()<br>
            break<br>
             
    for i, val in enumerate(c):<br>
        if t[i] == "Yes":<br>
            for x in range(len(specific_hypothesis)):<br>
                if val[x] != specific_hypothesis[x]:<br>
                    specific_hypothesis[x] = '?'<br>
                else:<br>
                    pass<br>
                return specific_hypothesis<br>
 
**obtaining the final hypothesis**<br>
print("The final hypothesis is:",train(d,target))<br>
*OUTPUT:*<br>
![image](https://user-images.githubusercontent.com/98145023/209805943-970cea7a-6766-4100-9a3c-68f874bd9de1.png)<br><br><br>



# Write a program to implement the Candidate-Elimination algorithm, For a given set of training data examples stored in a .CSV file.<br>
import csv<br>
with open("Train.csv") as f:<br>
    csv_file=csv.reader(f)<br>
    data=list(csv_file)<br>
    s=data[1][:-1]<br>
    g=[['?' for i in range(len(s))] for j in range(len(s))]<br>
    for i in data:<br>
        if i[-1]=="Yes":<br>
            for j in range(len(s)):<br>
                if i[j]!=s[j]:<br>
                    s[j]='?'<br>
                    g[j][j]='?'<br>
        elif i[-1]=="No":<br>
            for j in range(len(s)):<br>
                if i[j]!=s[j]:<br>
                     g[j][j]=s[j]<br>
                else:<br>
                    g[j][j]="?"<br>
        print("\nSteps of Candidate Elimination Algorithm",data.index(i)+1)<br>
        print(s)<br>
        print(g)<br>
    gh=[]<br>

    for i in g:
        for j in i:<br>
            if j!='?':<br>
                gh.append(i)<br>
                break<br>
print("\nFinal specific hypothesis:\n",s)<br>
print("\nFinal general hypothesis:\n",gh)<br>
*OUTPUT:*<br>
![image](https://user-images.githubusercontent.com/98145023/210222450-a3f9539a-fe89-4a1b-9235-98f89804d1a8.png)<br><br><br>



# Write a Program to Implement N-Queens Problem using Python.<br>
global N<br>
N = 4<br>
def printSolution(board):<br>
    for i in range(N):<br>
        for j in range(N):<br>
            print (board[i][j], end = " ")<br>
        print()<br>

def isSafe(board, row, col):<br>
    for i in range(col):<br>
        if board[row][i] == 1:<br>
            return False<br>
    for i, j in zip(range(row, -1, -1),range(col, -1, -1)):<br>
        if board[i][j] == 1:<br>
            return False<br>
    for i, j in zip(range(row, N, 1),range(col, -1, -1)):<br>
        if board[i][j] == 1:<br>
            return False<br>
    return True<br>

def solveNQUtil(board, col):<br>
    if col >= N:<br>
        return True<br>
    for i in range(N):<br>
        if isSafe(board, i, col):<br>
            board[i][col] = 1<br>
            if solveNQUtil(board, col + 1) == True:<br>
                return True<br>
            board[i][col] = 0<br>
    return False<br>

def solveNQ():<br>
    board = [ [0, 0, 0, 0],<br>
              [0, 0, 0, 0],<br>
              [0, 0, 0, 0],<br>
              [0, 0, 0, 0] ]<br>
    if solveNQUtil(board, 0) == False:<br>
        print ("Solution does not exist")<br>
        return False<br>
    printSolution(board)<br>
    return True<br>

solveNQ()<br>
*OUTPUT:*<br>
![image](https://user-images.githubusercontent.com/98145023/210223012-85598c5e-61cb-49c9-9c60-340ca431fd9a.png)<br><br><br>


# Write a Program to Implement A* algorithm using Python.<br>
def aStarAlgo(start_node, stop_node):<br>
         
        open_set = set(start_node) <br>
        closed_set = set()<br>
        g = {} #store distance from starting node<br>
        parents = {}# parents contains an adjacency map of all nodes<br>
 
        **ditance of starting node from itself is zero**<br>
        g[start_node] = 0<br>
        **start_node is root node i.e it has no parent nodes**so start_node is set to its own parent node<br>
        parents[start_node] = start_node<br>
         
         
        while len(open_set) > 0:<br>
            n = None<br>
 
          **node with lowest f() is found**<br>
            for v in open_set:<br>
                if n == None or g[v] + heuristic(v) < g[n] + heuristic(n):<br>
                    n = v<br>
             
                     
            if n == stop_node or Graph_nodes[n] == None:<br>
                pass<br>
            else:<br>
                for (m, weight) in get_neighbors(n):<br>
                    **nodes 'm' not in first and last set are added to first**<br>
                    #n is set its parent<br>
                    if m not in open_set and m not in closed_set:<br>
                        open_set.add(m)<br>
                        parents[m] = n<br>
                        g[m] = g[n] + weight<br>
                         
     
                   **for each node m,compare its distance from start i.e g(m) to the**<br>
                    **from start through n node**<br>
                    else:
                        if g[m] > g[n] + weight:<br>
                            #update g(m)<br>
                            g[m] = g[n] + weight<br>
                            #change parent of m to n<br>
                            parents[m] = n<br>
                             
                            **if m in closed set,remove and add to open**<br>
                            if m in closed_set:<br>
                                closed_set.remove(m)<br>
                                open_set.add(m)<br>
 
            if n == None:<br>
                print('Path does not exist!')<br>
                return None<br>
 
            ** if the current node is the stop_node**<br>
           ** then we begin reconstructin the path from it to the start_node**<br>
            if n == stop_node:<br>
                path = []<br>
 
                while parents[n] != n:<br>
                    path.append(n)<br>
                    n = parents[n]<br>
 
                path.append(start_node)<br>
 
                path.reverse()<br>
 
                print('Path found: {}'.format(path))<br>
                return path<br>
 
 
          ** remove n from the open_list, and add it to closed_list**<br>
           ** because all of his neighbors were inspected**<br>
            open_set.remove(n)<br>
            closed_set.add(n)<br>
 
        print('Path does not exist!')<br>
        return None<br>
         
**define fuction to return neighbor and its distance**<br>
**from the passed node**<br>
def get_neighbors(v):<br>
    if v in Graph_nodes:<br>
        return Graph_nodes[v]<br>
    else:<br>
        return None<br>
**for simplicity we ll consider heuristic distances given**<br>
**and this function returns heuristic distance for all nodes**<br>
def heuristic(n):<br>
        H_dist = {<br>
            'A': 11,<br>
            'B': 6,<br>
            'C': 99,<br>
            'D': 1,<br>
            'E': 7,<br>
            'G': 0,<br>
             
        }<br>
 
        return H_dist[n]<br>
 
**Describe your graph here <br> 
Graph_nodes = {<br>
    'A': [('B', 2), ('E', 3)],<br>
    'B': [('C', 1),('G', 9)],<br>
    'C': None,<br>
    'E': [('D', 6)],<br>
    'D': [('G', 1)],<br>
     
}<br>
aStarAlgo('A','G')<br>
*OUTPUT:*<br>
![image](https://user-images.githubusercontent.com/98145023/210535733-c331764e-e7f8-4a8d-8886-63e3bbec85ff.png)<br><br><br>
