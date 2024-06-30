import numpy as np
import pandas as pd
import sys
from collections import deque
import heapq
import copy 
import random


INF=sys.maxsize

DEFAULT_GRAPH = {  
        "A": ["X", "Y", "C"],
        "B": ["A", "Y"],
        "X": ["Y", "C" , "B"],
        "C": ["B", "Y", "Z"],
        "Y": ["A", "Z"],	
        "Z": ["B"]	}

DEFAULT_WEIGHTED_GRAPH={
    "A": {"X": random.randint(1,7), "Y": random.randint(1,7), "C": random.randint(1,7)},
    "B": {"A": random.randint(1,7), "Y": random.randint(1,7)},
    "X": {"Y": random.randint(1,7), "C": random.randint(1,7), "B": random.randint(1,7)},
    "C": {"B": random.randint(1,7), "Y": random.randint(1,7), "Z": random.randint(1,7)},
    "Y": {"A": random.randint(1,7), "Z": random.randint(1,7)},
    "Z": {"B": random.randint(1,7)}
    }

DEFAULT_WEIGHTED_GRAPH_R={
    "A": {"X": random.randint(-7,7), "Y": random.randint(-7,7), "C": random.randint(-7,7)},
    "B": {"A": random.randint(-7,7), "Y": random.randint(-7,7)},
    "X": {"Y": random.randint(-7,7), "C": random.randint(-7,7), "B": random.randint(-7,7)},
    "C": {"B": random.randint(-7,7), "Y": random.randint(-7,7), "Z": random.randint(-7,7)},
    "Y": {"A": random.randint(-7,7), "Z": random.randint(-7,7)},
    "Z": {"B": random.randint(-7,7)}
    }


def createWordedNumbering(n: int) -> list[str]:
    arr: list[str] = [""]*n
    for i in range(1,n+1):
        match i:
            case 1: arr[i-1]= "first"
            case 2: arr[i-1]= "second"
            case 3: arr[i-1]= "third"
            case _: arr[i-1]= f"{i}th"
    return arr

def createGraph(n=-1):
    if n==-1: 
        print ("Enter the number of nodes you want in the graph:")
        n= int(input())
    numbers= createWordedNumbering(n)
    AdjList = {}
    for vertex in range(n):
        print(f"What is the name of the {numbers[vertex]} node?")
        Nname= input()
        AdjList[Nname]= []
        print (f"How many nodes does node '{Nname}' point to?:")
        m= int(input())
        for connection in range(m):
            print ("Enter the neighbor:")
            neighbor = input()
            AdjList[Nname].append(neighbor)
    
    return AdjList
  
import matplotlib.pyplot as plt
import networkx as nx

def visualize_graph(graph: dict ) -> None:  #GPT generated
  """
  This function visualizes a graph represented as a dictionary.

  Args:
      graph: A dictionary where keys are node names and values are lists of neighboring nodes.
  """
  # Create a NetworkX graph object
  G = nx.DiGraph()  # Use DiGraph for directed graphs (if needed)

  # Add nodes to the graph
  for node, neighbors in graph.items():
    G.add_node(node)

  # Add edges to the graph
  for node, neighbors in graph.items():
    for neighbor in neighbors:
      G.add_edge(node, neighbor)

  # Set node positions (optional)
  pos = nx.spring_layout(G)  # Use a layout algorithm (like spring_layout)

  # Draw the graph
  nx.draw_networkx_nodes(G, pos, node_color='red', node_size=500)
  nx.draw_networkx_edges(G, pos, alpha=0.7)

  # Add node labels
  nx.draw_networkx_labels(G, pos, font_size=16)

  # Show the graph
  plt.axis('off')  # Hide unnecessary axis labels
  plt.show()

#-------------------------------------------------------------------------------


def BFS(graph: dict, start: str, anotate: bool = False, info: bool = True) -> tuple[dict,dict]: #Returns two dictionaries, one for distances and one for predecessors.


    #Step 1- Initialize the variables
    queue =deque() # Empty queue 
    distance= {i: INF for i in graph} #Initializing a dicitionary of the nodes in the graph with their values set to infinity to resmble undiscovered/unreachable nodes.
    distance[start]=0 #Initializing the starting node distance value to 0
    lead= {i: [None] for i in graph} #Initilazing an empty list in the size of the nodes in the graph to act as 
    
    #Step 2- Begin BFS from start node
    queue.append(start)
    while queue:
        vertex= queue.popleft() #Popping the first node in the queue
        if anotate: print("Visiting: " + vertex)
        if anotate: print(f"Neighbors of {vertex}: {graph[vertex]}")
        for connection in graph[vertex]: #For every node in the neighbors list
            if distance[connection]==INF: #If undiscovered:
                distance[connection]= distance[vertex]+1 #Discovery: Update the distance.
                lead[connection] = vertex #Discovery: Update the path.
                queue.append(connection) #Discovery: Add reacheable node to the queue.

    if info: print("The distance from the starting node to each node is: ", distance)
    if info: print("The predecessor array is: ", lead)

    return distance, lead

def DFS(graph: dict, anotate: bool = False) -> tuple[dict , dict , bool]: #Returns two dictionaries, one for distances and one for predecessors, in addition to a flag for a cycle

    #Step 1- Initialize the variables
    cycle: bool = False

    lead = {i: [None] for i in graph} #Predecesors dicitionary
    start = {i:INF for i in graph} #Discovery dicitionary
    finish = {i:INF for i in graph} #Finish dicitionary
    time= {"start":start, "finish":finish}

    tick=1 # - Node operation ticks - AKA time/t

    #Definining nested DFS_rec function that will live only while DFS is running.
    def DFS_rec(vertex: str) -> None:
        #Declaring the fact that these variables are the same for both DFS and DFS_rec, in order to use them in DFS_rec
        nonlocal start, finish, tick, lead, cycle, anotate      
         
        if anotate: print("Visting " + vertex + " at time: " , tick)
        start[vertex]=tick #Setting the discovery time of the node.
        tick=tick+1 #Advancing time by 1.

        for neighbor in graph[vertex]: #For every neighbor...
            if start[neighbor]!=INF and finish[neighbor]==INF: #If the neighbor was discovered and not finished AKA color is grey (status processing)
                if anotate: print("Cycle detected. There is a back edge between ", vertex, " and ", neighbor)
                cycle= True
            if start[neighbor]==INF: #If the neighbor was not yet discovered
                lead[neighbor]= vertex #Setting predecesor
                DFS_rec(neighbor)
        
        if anotate: print("Finishing with " + vertex + " at time: " , tick)
        finish[vertex]=tick #Setting finish time for processing.
        tick=tick+1 #Advancing time by 1.
        return


    for vertex in graph:  #For every vertex in the graph
        if start[vertex]==INF: #If undiscovered
            DFS_rec(vertex)
    


    print("The starting/finishing times for each vertex are: ")
    for key in graph :
        print(key, " starts at: ",start[key], " and finishes at: ", finish[key])
    print("The predecsor array is: ", lead)
    
    return time,lead,cycle

#-------------------------------------------------------------------------------

def removeNodes(graph: dict, A: list) -> dict:
    
    graph_new=copy.deepcopy(graph)

    for vertex in A:
        del graph_new[vertex]
    for key in graph_new:
        for vertex in A:
            if vertex in graph_new[key]: graph_new[key].remove(vertex)
    return graph_new
    
def copyGraph(graph: dict) -> dict:
    graph_new= copy.deepcopy(graph)

    return graph_new

def Transpose_Graph(graph: dict) -> dict:
    new_graph = {}
    for key in graph:
        for neighbor in graph[key]:
            if neighbor not in new_graph:
                new_graph[neighbor]= [key]
            else:
                new_graph[neighbor].append(key)
    return new_graph

def SCC(graph: dict)-> dict:
    SCC_groups={}
    group_index=1
    tgraph= Transpose_Graph(graph)
    time,lead,cycle= DFS(graph)
    fin=time["finish"]
    fin = dict(sorted(fin.items(), key= lambda item : item[1], reverse= True)) #Grabbing the tuples from the dictionary, then sorting them using the sorted() with the key lambda function to extract the value from the tuples, then use the dict function to turn it back to dictionary.

    def miniDFS(node: str) -> None:
        nonlocal group_index, SCC_groups
        if node not in SCC_groups:
            SCC_groups[node]= group_index
            print ("Node " + node, "belongs to group " , group_index)

            for neighbor in tgraph[node]:
                miniDFS(neighbor)


    for node in fin:
        if node not in SCC_groups:
            miniDFS(node)
            group_index+=1

    return SCC_groups
            

#-------------------------------------------------------------------------------

def createWeightedGraph(n: int = -1) -> dict:
    if n==-1:
        print ("Enter the number of nodes you want in the graph:")
        n= int(input())
    numbers= createWordedNumbering(n)
    graph = {}
    for vertex in range(n):
        print(f"What is the name of the {numbers[vertex]} node?")
        Nname= input()
        graph[Nname]= []
        print (f"How many nodes does node '{Nname}' point to?")
        m= int(input())
        for connection in range(m):
            print ("Enter the neighbor:")
            neighbor = input()
            print ("Enter the weight:")
            w= int(input())
            graph[Nname].append((neighbor, w)) 
    return graph 

def Dijkstra(graph: dict, start: str, directed: bool =True, info: bool =False) -> tuple[dict,dict]:
    #-----------------------Intilization steps-----------------------
    distances, lead
    condensed= True
    vSize= len(graph)
    eSize= sum(len(graph[i]) for i in graph )
    if info: print("Number of nodes is ", vSize, "and number of edges is ", eSize if directed else eSize/2)

    #----Determine if the graph is condenesed.
    if directed:
        if vSize*vSize > eSize:
            condensed= False
            if info: print("Set condensed to false.")
    else:
        if vSize*vSize > (eSize/2):
            condensed= False
            if info: print("Set condensed to false.")
    
    #------------------------Diksjtra with heap implementation
    def DijkstraSparse() -> None:
        nonlocal  info , graph, start #Grabbing parameters from parent function
        nonlocal distances, lead #Grabbing lead and distances dictionarys from parent function
        distances= {}
        lead = {}

        Q_to_visit= [(0, start)] #Entering the start node to the queue with the distance 0.
        lead[start]= None
        while Q_to_visit: #While queue is not empty:
            dist, vertex = heapq.heappop(Q_to_visit) #Grabbing the node with the smallest distance value from the heap.
            print("Removed node:",vertex,"from the queue with distance ",dist,". New queue is:", Q_to_visit) if info else None
            if vertex in distances:
                print("Node", vertex, "has already set it's' value in the distances list. Skipping..") if info else None
                continue
            distances[vertex]=dist #Setting the value for the vertex in the distances list.
            print("Assigned vertex:", vertex , "with its final distance value of", dist) if info else None
            print("Updated distances list: ", distances) if info else None
            for neighbor, weight in graph[vertex]: #For each neighbor..
                if neighbor not in distances: #If the neighbor has not yet had it's final value set..
                    print("Appending vertex", neighbor, "to the queue with the distance of ", dist+weight) if info else None
                    heapq.heappush(Q_to_visit, (dist+weight, neighbor))
                    if (neighbor in lead) & (dist+weight < Q_to_visit[[i for i, (_, t) in enumerate(Q_to_visit) if t==neighbor][0]][0]) : #Checking if the new distance is less than the old distance in the heap.
                        print("Preforming relaxation...") if info else None
                        lead[neighbor]= vertex
                        print("Preformed relaxation. New predecesor for", neighbor, "is", vertex, "and it's new set distance is", dist+weight+".", "New queue is: ", Q_to_visit) if info else None
                    else: 
                        lead[neighbor]= vertex
                        if neighbor not in lead: print("New predecesor for", neighbor, "is", vertex, "and it's new set distance is", dist+weight+".", "New queue is: ", Q_to_visit) if info else None
    if condensed == False: 
        DijkstraSparse(graph, start)
        
    #------------------------Dijkstra with array implementation
    else: 
        S_visited= [] #Empty visited list
        distances= {i: INF for i in graph} #Intilization of distance array to infinity
        lead = {i: None for i in graph} #Intilization of predecessor array to NULL
        Q_to_visit= [i for i in graph] #Creating a list of all the nodes in the graph
        distances[start]= 0 #Base case node. Setting distance from itself to itself as 0.



        while Q_to_visit:
            #-------------------See if this loop works to find minimum value of distance so far----------------
            minVal= INF #Assigning/reassigning the minimum value to infinity to avoid last iteration's assignment.
            removeMe= None
            for key in graph:
                if (minVal> distances[key]) & (key not in S_visited):
                    minVal= distances[key]
                    removeMe=key
            print("for loop found the node with the minimum value of distance so far and it is: ", removeMe, " With the value: ", distances[removeMe]) if info else None  #Optional Print
           
            #--------------------------------------------------------------------------------------------------
            vertex= Q_to_visit.pop(Q_to_visit.index(removeMe)) #Dequeuing the node with the minimum value using the loop to find the minimum value.
            print("Removed node: ", vertex, " the new Q_to_visit is: ", Q_to_visit) if info else None #Optional Print
            S_visited.append(vertex) #Adding the node to the visited list
            for neighbor, weight in graph[vertex]: #For every neighbor..
                if neighbor not in S_visited: #If the neighbor was not yet visited..
                    if distances[neighbor] > distances[vertex] + weight: #Relax phase, check if the new distance is shorter than what's already applied.
                        print("Preforming relaxation... ") if info else None #Optional Print
                        distances[neighbor]= distances[vertex] + weight
                        lead[neighbor]= vertex #Setting the new predecesor.
                        print("Preformed relaxtion, the new distance is: ", distances[neighbor], " and the new predecesor for node ", neighbor, " is: ", vertex) if info else None #Optional Print
    if info:
        for key in distances:
            print("The distance from node " +start+ " to the node " +key+ " is: " +str(distances[key]) ) #Optional Print
            print("The predecesor value for this node is: " + str(lead[key]) )  #Optional Print
        
        print("The predecsor array in full is: ", lead) #Optional Print

    return distances, lead

def BelmanFord(graph, start):
    pass


#-------------------------------------------------------------------------------

def GCD(x, y):
    temp = 0
    while(y != 0):
        temp = y
        y = x % y
        x = temp

    return x



class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        if self.head is None:
            self.head= Node(data)
        else:
            last_node= self.head
            while last_node.next:
                last_node= last_node.next
            last_node.next= Node(data)
    
    def display(self):
        current= self.head
        while current:
            print(current.data,end=' -> ')
            current = current.next
        print ("None")

class AVLTree:
    """
    AVL Tree implementation in Python
    
    Works with integers only.
    """


    def __init__(self,*, data:int) -> None:
        self.data: int =data
        self.balance_factor: int=0
        self.height: int=1
        self.left: 'AVLTree' = None
        self.right: 'AVLTree' = None
        self.parent: 'AVLTree' = None


    def addNode(self, *, value: int) -> None:

        print("Visiting node: ", self.data)

        if value < self.data: #RUN LEFT
            if self.left is None:
                self.left = AVLTree(data=value)
                self.left.parent = self
            else:
                self.left.addNode(value=value)
        else: #RUN RIGHT
            if self.right is None:
                self.right = AVLTree(data=value)
                self.right.parent = self
            else:
                self.right.addNode(value=value)
        
        self.calculateBalance()
        self.fixTree()

    def removeNode(self,*, value: int) -> None:
        workNode: 'AVLTree' = self.find(value)
        foreparent: 'AVLTree' = workNode.parent
        traverse: 'AVLTree'= None

        if workNode is None:
            return
        

        if foreparent is None: #Case - Root
            if workNode.balance_factor > 0:
                traverse = workNode.right
                while traverse.left is not None: # Getting to the minimum of the right branch of tree
                    traverse = traverse.left
                traverse.left = workNode.left
                workNode.left.parent = traverse
                workNode.right.parent = None
                self.fixTree(workNode.left)
                del workNode

            elif workNode.balance_factor < 0:
                traverse = workNode.left
                while traverse.right is not None: # Getting to the maximum of the left branch of tree
                    traverse = traverse.right
                traverse.right = workNode.right
                workNode.right.parent = traverse
                workNode.left.parent = None
                self.fixTree(workNode.right)
                del workNode
                

        elif foreparent.left is workNode: #Case - Left Child
            traverse = workNode.right
            if traverse is not None:
                while traverse.left is not None: # Getting to the minimum of the right branch of tree
                    traverse = traverse.left
            else:
                traverse = workNode.left

            traverse, workNode = workNode, traverse
            del traverse
            foreparent.left = workNode
            if workNode is not None:
                AVLTree.fixTree(workNode)
        
        elif foreparent.right is workNode: #Case - Right Child
            traverse = workNode.left
            if traverse is not None:
                while traverse.right is not None: # Getting to the maximum of the left branch of tree
                    traverse = traverse.right
            else: 
                traverse = workNode.right
            
            traverse, workNode = workNode, traverse
            del traverse
            foreparent.right = workNode
            if workNode is not None:
                AVLTree.fixTree(workNode)

        foreparent.calculateBalance() 

    def find(self, key: int) -> 'AVLTree':
        if key == self.data:
            return self
        
        if key < self.data:
            if self.left is None:
                return None
            else:
                return self.left.find(key)
        
        if key > self.data:
            if self.right is None:
                return None
            else:
                return self.right.find(key)
        
        else:
            return None
        
    def fixTree(workNode: 'AVLTree') -> None:
        traverse= workNode

        #If a balance factor is 2 or -2, we need to fix the tree along the path from the node to the root.

        while traverse.parent is not None:

            #Recaulculating balance factors and heights:
            traverse.calculateBalance()
            #print("I break here!")
            #Fixing tree using rotations corresponding to balance factor:
            match traverse.balance_factor:
                case -2:
                    if traverse.right.balance_factor == 1:
                        AVLTree.rotateRight(traverse.right) #Case -2, 1: Break on purpose to create -2,-1 then fix by rotation left.
                    AVLTree.rotateLeft(traverse) #Case -2, -1: Rotate left
                    
                case 2:
                    if traverse.left.balance_factor == -1:
                        AVLTree.rotateLeft(traverse.left) #Case 2, -1: Break on purpose to create 2,1 then fix by rotation right.
                    AVLTree.rotateRight(traverse) #Case 2, 1: Rotate right
            
            #print(f"traverse before = {traverse}")        
            traverse=traverse.parent
            #print(f"traverse after = {traverse}")        
            
    def rotateLeft(workNode: 'AVLTree') -> None:
        #A
        #B
        #B.parent = A.parent
        #B.parent.right = B
        #A.right = B.left
        #B.left.parent = A
        #B.left = A
        #A.parent = B

        oldFather_leftSon: 'AVLTree' = workNode #The node we encountered with bad balance factor. It is the oldFather, and after the process it'll be the left son.
        rightSon_newFather: 'AVLTree' = workNode.right #The pivit node, the node we preform the rotation on. It is the right Son, and after the process it'll be the newFather.

        rightSon_newFather.parent = oldFather_leftSon.parent #Breaking the link with old father, making the parent field be the parent of the old father
        rightSon_newFather.parent.right = rightSon_newFather #Making sure the link is connected from forefather side.

        
        oldFather_leftSon.right = rightSon_newFather.left #Transfering all the smaller(left) kids of the new father to the old fathers bigger(right) kids.
        if rightSon_newFather.left is not None:
            rightSon_newFather.left.parent= oldFather_leftSon # SAME AS A LINE ABOVE


        rightSon_newFather.left = oldFather_leftSon #Making the old father the left son of the new father.
        oldFather_leftSon.parent = rightSon_newFather #Updating the new father to be the parent of the old father.

        oldFather_leftSon.calculateBalance() #Recalculating the balance factor of the old father
        rightSon_newFather.calculateBalance() #Recalculating the balance factor of the new father

    def rotateRight(workNode: 'AVLTree') -> None:
        
        #A
        #B
        #B.parent = A.parent
        #B.parent.left = B
        #A.left = B.right
        #B.right.parent = A
        #B.right = A
        #A.parent = B
        
        oldFather_rightSon: 'AVLTree' = workNode #The node we encountered with bad balance factor. It is the oldFather, and after the process it'll be the right son.
        leftSon_newFather: 'AVLTree' = workNode.left #The pivit node, the node we preform the rotation on. It is the right Son, and after the process it'll be the newFather.

        leftSon_newFather.parent = oldFather_rightSon.parent #Breaking the link with old father, making the parent field be the parent of the old father
        leftSon_newFather.parent.left = leftSon_newFather #Making sure the link is connected from forefather side.
        
        oldFather_rightSon.left = leftSon_newFather.right #Transfering all the bigger(right) kids of the new father to the old fathers smaller(left) kids.
        if leftSon_newFather.right is not None:
            leftSon_newFather.right.parent = leftSon_newFather # SAME AS A LINE ABOVE

        leftSon_newFather.right = oldFather_rightSon #Making the old father the left son of the new father.
        oldFather_rightSon.parent = leftSon_newFather #Updating the new father to be the parent of the old father.

        #Updating...:
        oldFather_rightSon.calculateBalance()
        leftSon_newFather.calculateBalance()

    def calculateBalance(self):
    
        match self.left, self.right:
            case None, None:
                self.balance_factor= 0
                self.height = 1  
            case None, _:
                self.balance_factor= -self.right.height
                self.height = 1 + self.right.height
            case _, None:
                self.balance_factor= self.left.height
                self.height = 1 + self.left.height
            case _, _:
                self.balance_factor= self.left.height - self.right.height
                self.height = 1 + max(self.left.height, self.right.height)
    
    
    def inOrder(self) -> None:
        
        if self.left is not None:
            self.left.inOrder()
        
        print(self.data, end = ", ")

        if self.right is not None:
            self.right.inOrder()
    
    def preOrder(self) -> None:

        print(self.data, end = ", ")
        
        if self.left is not None:
            self.left.preOrder()
        

        if self.right is not None:
            self.right.preOrder()

    
    def postOrder(self) -> None:

        if self.left is not None:
            self.left.postOrder()
        

        if self.right is not None:
            self.right.postOrder()

        print(self.data, end = ", ")

        
    
    def __repr__(self) -> str:
        return f"(data={self.data},balance_factor={self.balance_factor},height={self.height})"


def visualizeGraph_alternate() -> None:
    graph_dict: dict[str:list[str]] = {
        "A": ["X", "Y", "C"],
        "B": ["A", "Y"],
        "X": ["Y", "C" , "B"],
        "C": ["B", "Y", "Z"],
        "Y": ["A", "Z"],	
        "Z": ["B"]	
    }

    Graph: nx.Graph= nx.Graph(graph_dict)

    pos = nx.shell_layout(Graph)
    nx.draw(Graph, pos, with_labels=True,
            node_color= 'lightblue',
            node_size=500, font_size=10,
            font_weight="bold",
            edge_color="grey",
            edge_width= 2,
            edge_style="arrowed"
            )
    
    plt.show()

#region Assignment 1.1 <---------------------->

def Walk_Through_Node(graph, x, z, y): #Undirected Graph
    d, lead= BFS(graph, z)
    return True if (d[x]!=INF) and (d[y]!=INF) else False

def Path_Through_Node(graph, x, z, y): #Undirected Graph
    d, lead= BFS(graph, x)
    if d[y]==INF:
        return False
    else: dis= d[y]
    
    d, lead = BFS(graph, z)
    return False if dis!=(d[y]+d[x]) else True


def Shortest_Walk_Through_Node_Val(graph, x, z, y): #Undirected Graph
    d, lead= BFS(graph, z)
    return d[y]+d[x] if (d[x]!=INF) and (d[y]!=INF) else INF


def Walk_Through_Group(graph,x,A,y): #Undirected Graph
    d, lead= BFS(graph, x)
    if d[y]==INF: return False
    for vertex in A:
        if d[vertex]!=INF: return True
    return False


#-----------------------GROUPS--------------------------


def Path_Through_Group(graph,x,A,y): #Undirected Graph
    d, lead= BFS(graph, x)
    if (d[y]==INF) or d[y]==1: return False
    else: dis= d[y]
    
    graph_new= removeNodes(graph, A)
    d, lead = BFS(graph_new, x)
    return False if d[y]<= dis else True


def Shortest_Walk_Through_Group_Val(graph: dict, x: str, A: list[str], y : str) -> int: #Undirected Graph
    dx, leadx= BFS(graph, x)
    if (dx[y]==INF): return INF

    dy, leady= BFS(graph, y)

    min=INF
    for vertex in A:
        if(dx[vertex]+dy[vertex] < min):
            min = dx[vertex]+dy[vertex]
    return min


def Merge_Graphs_shared_nodes(graph1: dict, graph2: dict) -> dict: #Question: What will happen when the two graphs don't have the same nodes?
    graph_new= {}
    for key in graph1:
        graph_new[key]= graph1[key] #Step 1- Copy all of the nodes and the edges from the first graph.
    for key in graph2:
        if key not in graph1: #Step 2- Begin copying nodes from graph2. If the nodes are not in graph 1, we append the whole list of neighbors AKA the nodes + the edges that node participates in.
            graph_new[key]=graph2[key]
        else:
            for neighbor in graph2[key]:  #Step 3- If the nodes are in graph 1, we append only the neighbors AKA the edges that are not already in graph 1.
                if neighbor not in graph1[key]:
                    graph_new[key].append(neighbor)

    print (graph_new) #Step 4- Print the new graph- confirimation.
    return graph_new

#-----------------------TOPOLOGY--------------------------

def Detect_Same_topology(graph1: dict, graph2: dict) -> bool:
    graph_new= Merge_Graphs_shared_nodes(graph1, graph2)
    d,lead,cycle= DFS(graph_new)
    return False if cycle else True


#-----------------------TESTS----------------------------

def suggest(prompt: str) -> bool:
    ans: int
    print(prompt)
    ans = int(input())
    return True if ans == 1 else False

def test_Assign1_1(interactive: bool = False, graph: dict = None) -> bool:
    
    if graph == None:
        if suggest("Would you like to use the default graph or create a new one? (0=default, 1=create):")== True:
            graph= createGraph()
        else:
            graph = DEFAULT_GRAPH

    if interactive:
        print("Where would you like to run BFS from?")
        d, lead= BFS(graph, input())
        print("What path are you looking to validate?")
        x:str= input("Path from ").strip()
        y:str= input(" to ").strip()
        z:str= input(" through ").strip()
        ans= Walk_Through_Node(graph, x, z, y)
        print (f"There is a walk between {x} and {y} that passes through {z}." if ans else f"There is no walk between {x} and {y} that passes through {z}.")


    else:
        d, lead= BFS(graph, 'X')
        ans= Walk_Through_Node(graph, 'X', 'Z', 'Y')
        print ("There is a walk between x and y that passes through z." if ans else "There is no walk between x and y that passes through z.")


    

    return suggest("Would you like to continue or repeat? (0=continue, 1=repeat):")
        
def test_Assign1_2(interactive: bool = False, graph: dict = None) -> bool:
    
    if graph == None:
        if suggest("Would you like to use the default graph or create a new one? (0=default, 1=create):")== True:
            graph= createGraph()
        else:
            graph = DEFAULT_GRAPH

    if interactive:
        print("Which path are you looking to validate?")
        x= input("Path from ").strip()
        y= input(" to ").strip()
        z= input(" through ").strip()
        ans =Path_Through_Node(graph,x,z,y)
        print(f"There is a path between {x} and {y} that passes through {z}." if ans else f"There is no path between {x} and {y} that passes through {z}.")	


    else:
        ans = Path_Through_Node(graph, 'X', 'Z', 'Y')
        print ("There is a path between x and y that passes through z." if ans else "There is no path between x and y that passes through z.")


    

    return suggest("Would you like to continue or repeat? (0=continue, 1=repeat):")
    
def test_Assign1_3(interactive: bool = False, graph: dict = None) -> bool:
    
    if graph == None:
        if suggest("Would you like to use the default graph or create a new one? (0=default, 1=create):")== True:
            graph= createGraph()
        else:
            graph = DEFAULT_GRAPH

    if interactive:
        print("Which walk are you looking to find the min for?")
        x= input("Path from ").strip()
        y= input(" to ").strip()
        z= input(" through ").strip()
        ans= Shortest_Walk_Through_Node_Val(graph, z, x, y)


    else:
        ans= Shortest_Walk_Through_Node_Val(graph, 'X', 'Z', 'Y')
    
    print ("The value of the shortest walk between x and y that passes through z is: ", ans)

    

    return suggest("Would you like to continue or repeat? (0=continue, 1=repeat):")
    
def test_Assign2_1(interactive: bool = False, graph: dict = None) -> bool:
    
    if graph == None:
        if suggest("Would you like to use the default graph or create a new one? (0=default, 1=create):")== True:
            graph= createGraph()
        else:
            graph = DEFAULT_GRAPH

    if interactive:
        print("What is group A?:")
        A= input().split(sep=" " | ",") #group
        print("What is the path you would like to validate? (through group A.)(these nodes can't belong to A!):")
        x= input("Path from ").strip()
        y= input( " to ").strip()
        ans=Walk_Through_Group(graph, x, A, y)
        print(f"There is a walk between {x} and {y} that passes through group A." if ans else f"There is no walk between {x} and {y} that passes through group A.")
        ans= Path_Through_Group(graph, x, A, y)
        print (f"The shortest path between {x} and {y} indeed passes through group A." if ans else f"The shortest path between {x} and {y} does not pass through group A or is equal to the path that between {x} and {y} that passes through group A.") 
        ans=Shortest_Walk_Through_Group_Val(graph, x, A, y)
        print(f"The value of the shortest walk between {x} and {y} that passes through group A is: ", ans)

    else:
        A=['c','z']
        ans= Walk_Through_Group(graph, 'x', A, 'y')
        print ("There is a walk between x and y that passes through group A." if ans else "There is no walk between x and y that passes through group A.")
        ans= Path_Through_Group(graph, 'x', A, 'y')
        print ("The shortest path between x and y indeed passes through group A." if ans else "The shortest path between x and y does not pass through group A or is equal to the path that between x and y that passes through group A.")

        ans=Shortest_Walk_Through_Group_Val(graph, 'x', A, 'y')
        print("The value of the shortest walk between x and y that passes through group A is: ", ans)

    

    return suggest("Would you like to continue or repeat? (0=continue, 1=repeat):")

def test_Assign2_2(interactive: bool = False, graph: dict = None) -> bool:
    
    if graph == None:
        if suggest("Would you like to use the default graph or create a new one? (0=default, 1=create):")== True:
            graph= createGraph()
        else:
            graph = DEFAULT_GRAPH

    if interactive:
        Time, lead, cycle= DFS(graph)
        print("Now please enter graph 2 (where V= {A v belongs to V2, v belongs to V1}):")
        graph2=createGraph(len(graph))
        ans=Detect_Same_topology(graph,graph2)
        print ("The graphs have the same topology." if ans else "The graphs do not have the same topology.")


    else:
        d, lead, cycle= DFS(graph)
        graph2=createGraph(len(graph))
        ans=Detect_Same_topology(graph,graph2)
        print ("The graphs have the same topology." if ans else "The graphs do not have the same topology.")
    

    return suggest("Would you like to continue or repeat? (0=continue, 1=repeat):")

#endregion <---------------------->

def test_SCC() -> bool:

    graph:dict

    if suggest(0)== True:
        graph= createGraph()
    else:
        graph = DEFAULT_GRAPH

    SCC_groups= SCC(graph)

    print(SCC_groups,start="\nSCC groups are: ")

    return suggest(1)

def test_GraphVisualization() -> bool:
    graph:dict

    if suggest(0)== True:
        graph= createGraph()
    else:
        graph = DEFAULT_GRAPH

    visualize_graph(graph)


    return suggest("Would you like to continue or repeat? (0=continue, 1=repeat):")


def test_Dikstra(graph: dict = None, anotate: bool = False) -> bool:
    
    chngNode: bool = False

    if(graph == None):
        if suggest("Would you like to use the default graph or create a new one? (0=default, 1=create):")== True:
            graph= createWeightedGraph()
        else:
            graph = DEFAULT_WEIGHTED_GRAPH

    
    if anotate: print("graph: ",graph)

    while(chngNode== False):
        
        if chngNode == False: start= input("Please select a starting node: ")

        d, lead = Dijkstra(graph, start)
        if anotate: print("Predecesor array: ", lead, "distances list: ",d)

        chngNode = suggest("Would you like to change the starting node? (0=no, 1=yes): ")
    
    return suggest("Would you like to continue or repeat? (0=continue, 1=repeat):")
        
from collections import defaultdict

def countArray(text:str)-> dict:
    countArr = defaultdict(int)

    for char in text:
        countArr[char] += 1
    
    return countArr


from bitarray import bitarray

def hoffman_Encode(encodeMe: str) -> tuple[bitarray,dict]:
    countArr= countArray(encodeMe)
    heap = [[count, [char, '']] for char, count in countArr.items()] #Creating a list out of the div array dictionary.
    heapq.heapify(heap)
    #Pulling the letters with the least amount of instances from the heap

    while len(heap) > 1:
        left = heapq.heappop(heap) #Node with least amount of instances
        #print(f"Left = {left}")
        right = heapq.heappop(heap) #Second node with least amount of instances
        #print(f"Right = {right}")
        #Adding the values 0 or 1 to the selected branches, where 0 is left and 1 is right. We then add them together and call them by their new joint node name and sum.
        for pair in right[1:]:
            pair[1] = '1' + pair[1] #For example: ['l', '010'] becomes ['l', '1010']
        #print(f"right add one = {right}")
        for pair in left[1:]:
            pair[1] = '0' + pair[1]
        #print(f"left add zero = {left}")
        heapq.heappush(heap, [left[0] + right[0]] + left[1:] + right[1:]) #For example, [1,['l','1010']] and [2,['d','0010']] becomes [3, ['l', '1010'], ['d', '0010']]
    
    print(f"To the left: {left}\nAnd to the right: {right}")
    huffman_list = right[1:] + left[1:]
    #print(f"huffman_list = {huffman_list}")
    huffman_dict = {a[0]:bitarray(str(a[1])) for a in huffman_list}
    #print(f"huffman_dict = {huffman_dict}")

    encoded_text = bitarray()
    encoded_text.encode(huffman_dict, encodeMe)
    #print(encoded_text)

    return encoded_text, huffman_dict

def hoffmanIntoFile(encoding:bitarray, filename:str=None) -> int:
    if(filename is None):
        filename= input("Please enter a filename: ")
        filename = filename.strip() + ".bin"
       
    print(encoding)
    print("length of encoding: ", len(encoding))
    padding = 8 - (len(encoding) % 8)
    
    with open(filename, "wb") as f:
        encoding.tofile(f)
    
    return padding


def hoffmanFromFile(decoderDic:dict,filename:str=None, padding:int=0) -> str:

    if filename is None:
        filename= input("Please enter a filename (including .bin): ")
    
    if padding==0:
        padding = int(input("Please enter the padding: "))
    
    decodedRes = bitarray()

    with open (filename, "rb") as f:
        decodedRes.fromfile(f)
    
    decodedRes = decodedRes[:-padding]
    decodedRes = decodedRes.decode(decoderDic)
    decodedRes = "".join(decodedRes)

    return decodedRes




def main() -> None:
    
    res: bool = True
    #---------------------TESTING-------------------------
    #region <--> personal testing - tricks
    #-- List Comprehension:

    #arr = [key for key in graph if key!= value]
    #arr= {"1":[1,2], "2":[3,4], "3":[5,7]}
    #if (len(arr)*2)< sum (len(arr[i]) for i in arr):
    #    print("Yes")
    #else: print("No")

    #-- Heapq:
    # arr= [7,2,8,1,0,4,3,6,5]
    # heapq.heapify(arr)
    # print(arr)
    # t=heapq.heappop(arr)
    # print(arr)

    #-- Linked List:
    # mylist = LinkedList()
    # mylist.append(1)
    # mylist.append(2)
    # mylist.append(3)
    # mylist.display()

    # a: int = 5
    # b: int = 12
    # a,b = b,a
    # print(a,b)

    # myTree = AVLTree(data=5)
    # print("Hello World")
    # myTree.addNode(value=4)
    # myTree.addNode(value=6)
    # myTree.addNode(value=3)
    # myTree.addNode(value=7)
    # myTree.addNode(value=2)
    # myTree.addNode(value=10)
    # myTree.inOrder()
    # print("")
    # myTree.postOrder()
    # print("")
    # myTree.preOrder()
    # X=myTree.find(6)
    # print("")
    # print(X)

    # password = "HelloGithub9090"
    # encoded, keys = hoffman_Encode(password)
    # print(encoded)
    # print(len(encoded))
    # print(keys)

    # decoded_text = encoded.decode(keys)
    # decoded_text = "".join(decoded_text)
    # print(decoded_text)
    

    #endregion <-->

    #region <--> BFS, and assignment exercises testing.

    #---------------------BFS-----------------------------
    
    # while(res):
    #     res = test_Assign1_1(interactive = True)

    # while (res):
    #     res = test_Assign1_2(interactive = True)
    
    # while (res):
    #     res = test_Assign1_3(interactive = True)

    #---------------------DFS-----------------------------
    # while (res):
    #     res = test_Assign2_2(interactive = True)


    #endregion <-->

    #region SCC <--> SCC testing
    #---------------------SCC-----------------------------

    # while(res):
    #     res= test_SCC()
    # while(res):
    #     res= visualize_graph()

    #endregion <-->

    #region Dijsktra <-->
    #---------------------Dijkstra-----------------------------

    # while(res):
    #     res= test_Dikstra()

    #endregion <-->

    return

if __name__ == "__main__":
    main()