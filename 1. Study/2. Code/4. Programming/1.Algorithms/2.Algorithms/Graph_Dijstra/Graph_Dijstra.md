# 1. Dijkstra

- 최단거리 찾기 알고리즘

### 1) 우선순위 큐 구현 실습
- heapq 라이브러리 활용


```python
import heapq
```


```python
queue = []

heapq.heappush(queue,[2,'A'])
heapq.heappush(queue,[5,'B'])
heapq.heappush(queue,[1,'C'])
heapq.heappush(queue,[7,'D'])
print(queue)
```

    [[1, 'C'], [5, 'B'], [2, 'A'], [7, 'D']]



```python
for _ in range(len(queue)):
    print(heapq.heappop(queue))
```

    [1, 'C']
    [2, 'A']
    [5, 'B']
    [7, 'D']


### 2) 데이터 그래프 생성


```python
mygraph = {
    'A':{'B':8,'C':1,'D':2},
    'B':{},
    'C':{'B':5,'D':2},
    'D':{'E':3,'F':5},
    'E':{'F':1},
    'F':{'A':5},
}
```


```python
mygraph
```




    {'A': {'B': 8, 'C': 1, 'D': 2},
     'B': {},
     'C': {'B': 5, 'D': 2},
     'D': {'E': 3, 'F': 5},
     'E': {'F': 1},
     'F': {'A': 5}}



### 3) Dijkstra 구현


```python
import heapq

def dijkstra(graph,start):
    
    distance = {node:float('inf') for node in graph}
    
    distance[start] = 0
    queue = []
    heapq.heappush(queue,[distance[start],start])
    # queue안에는 기준 노드와 기준 노드의 거리(root로부터)가 들어있음
    
    while queue:
        current_distance,current_node = heapq.heappop(queue)
 
        for adjacent, weight in graph[current_node].items():
            distance_root_adjacent = current_distance + weight
            
            if distance_root_adjacent < distance[adjacent]:
                distance[adjacent] = distance_root_adjacent
                heapq.heappush(queue,[distance[adjacent],adjacent])

    return distance       
```


```python
dijkstra(mygraph,'A')
```




    {'A': 0, 'B': 6, 'C': 1, 'D': 2, 'E': 5, 'F': 6}




```python
import heapq

def dijkstra2(graph,start):
    distance = {node : float('inf') for node in graph}
    queue = []
    
    distance[start] = 0
    heapq.heappush(queue,[distance[start],start])
    
    # distance를 update할 차례. 반복해야겠지
    while queue:
        # current_distance = root - current까지 거리 
        current_distance,current_node = heapq.heappop(queue)
        
        for adjacent, weight in distance[current_node].items():
            distance_new = current_distance + weight
            
            if distance_new < distance[adjacent]:
                distance[adjacent] = distance_new
                heapq.heappush(queue,[distance[adjacent],adjacent])
            
    return distance
```


```python
dijkstra(mygraph,'A')
```




    {'A': 0, 'B': 6, 'C': 1, 'D': 2, 'E': 5, 'F': 6}


