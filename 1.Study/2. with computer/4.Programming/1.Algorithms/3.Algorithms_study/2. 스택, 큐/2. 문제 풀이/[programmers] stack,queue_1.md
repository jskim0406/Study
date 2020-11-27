# 1. 주식가격


[문제](https://programmers.co.kr/learn/courses/30/lessons/42584)

### 1) 1차


```python
def solution(prices):
    answer = []
    cnt = 0
    
    for i in range(len(prices)):
        for j in range(i, len(prices)-1):
            if prices[i] <= prices[j]:
                cnt += 1
                    
        answer.append(cnt)
        cnt = 0
    return answer
```


```python
solution([1, 2, 3, 2, 3])
```




    [4, 3, 1, 1, 0]



### 결과

- 10개 중 9개 case fail
- **시간 초과**

### 2) 2차

- 일단 시간 복잡도를 줄이기 위해, 주식가격이 떨어진 경우 for순환에서 break 하도록 코드 추가하는 수정


```python
def solution(prices):
    answer = []
    cnt = 0
    
    for i in range(len(prices)):
        for j in range(i, len(prices)-1):
            if prices[i] <= prices[j]:
                cnt += 1

            # 수정 부분
            else:
                break
                    
        answer.append(cnt)
        cnt = 0
    return answer
```


```python
solution([1, 2, 3, 2, 3])
```




    [4, 3, 1, 1, 0]



### 결과

- 성공
