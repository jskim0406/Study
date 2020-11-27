---
mathjax: true

title: 2. [프로그래머스 문제풀이] 3. H-index
categories:
- Algorithms
tags:
- python
- Algorithms
- 프로그래머스



---

# 3. H-index

[문제](https://programmers.co.kr/learn/courses/30/lessons/42747)

### 1) 1차 입력


```python
def solution(citations):
    sorted_cite = sorted(citations, reverse=True)
    
    answer = 0
    
    print(sorted_cite)
    
    for i in range(len(sorted_cite)):
        if sorted_cite[i] < i+1:
            answer = i
            return answer
```


```python
solution([3, 0, 6, 1, 5])
```

    [6, 5, 3, 1, 0]





    3




```python
solution([25,8,5,3,3])
```

    [25, 8, 5, 3, 3]





    3




```python
solution([25,8,5,4,3])
```

    [25, 8, 5, 4, 3]





    4



### 결과 : fail

- 9번 case 실패

### 2) 2차 입력


```python
def solution(citations):
    answer, max_idx = 0, 0
    sorted_cite = sorted(citations, reverse=True)
    
    for i in range(len(sorted_cite)):
        if sorted_cite[i] >= i+1:
            max_idx = i+1
    answer = max_idx
    return answer
```


```python
solution([3, 0, 6, 1, 5])
```




    3




```python
solution([25,8,5,3,3])
```




    3




```python
solution([25,8,5,4,3])
```




    4



### 결과 : 성공
