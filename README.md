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

