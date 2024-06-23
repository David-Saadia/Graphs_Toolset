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

#region Assignment 1.1

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


def Shortest_Walk_Through_Group_Val(graph,x,A,y): #Undirected Graph
    dx, leadx= BFS(graph, x)
    if (dx[y]==INF): return INF

    dy, leady= BFS(graph, y)

    min=INF
    for vertex in A:
        if(dx[vertex]+dy[vertex] < min):
            min = dx[vertex]+dy[vertex]
    return min


def Merge_Graphs_shared_nodes(graph1, graph2): #Question: What will happen when the two graphs don't have the same nodes?
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

def Detect_Same_topology(graph1, graph2):
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

#endregion

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

    while(res):
        res= test_SCC()
    while(res):
        res= visualize_graph()

    #endregion <-->

    #region Dijsktra <-->
    #---------------------Dijkstra-----------------------------

    while(res):
        res= test_Dikstra()

    #endregion <-->

    return

if __name__ == "__main__":
    main()