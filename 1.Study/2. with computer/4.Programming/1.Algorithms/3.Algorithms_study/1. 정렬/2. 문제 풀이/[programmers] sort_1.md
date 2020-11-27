---
mathjax: true

title: 2. [프로그래머스 문제풀이] 1. k번째수
categories:
- Algorithms
tags:
- python
- Algorithms
- 프로그래머스

---



# 1. k번째 수

[문제](https://programmers.co.kr/learn/courses/30/lessons/42748?language=python3)


```python
def solution(array, commands):
    answer = []
    
    for i in range(len(commands)):
        stand = commands.pop(0)
        sorted_array = sorted(array[stand[0]-1:stand[1]])
        answer.append(sorted_array[stand[2]-1])
    return answer
```
