# 2. 기능개발

[문제](https://programmers.co.kr/learn/courses/30/lessons/42586)

### 1) 1차


```python
def solution(progresses, speeds):
    answer = []
    days_list = []
    
    # 각 task별 소요일수 계산 후, task 우선순위대로 소요일수를 stack(List type 활용)
    for i in range(len(progresses)):
        days_left = 100-progresses[i]
        if days_left%speeds[i]==0:
            days_spent = days_left//speeds[i]
        else:
            days_spent = days_left//speeds[i] + 1
    
        days_list.append(days_spent)
        days_spent = 0

    
    # 소요일수가 "오르지 않는 날"의 지속기간 count("주식가격"문제와 동일한 구조)
    cnt, idx = 1, 0
    # stack이되, pop(0)으로 FIFO 구조로 활용
    temp = days_list.pop(0)
    while len(days_list) > idx:
        
        if temp >= days_list[idx]:
            cnt += 1
            days_list.pop(0)
        
            if len(days_list)<=1:
                answer.append(cnt)

        else:
            answer.append(cnt)
            cnt = 1
            temp = days_list.pop(0)
    
    return answer
```

### 결과

**[테스트 1] 에서 fail**
- 입력값 〉	[93, 30, 55], [1, 30, 5]
- 기댓값 〉	[2, 1]
- 실행 결과 〉	실행한 결괏값 [2]이(가) 기댓값 [2,1]와(과) 다릅니다.

### 2) 2차


```python
def solution(progresses, speeds):
    answer = []
    days_list = []
    
    # 각 task별 소요일수 계산 후, task 우선순위대로 소요일수를 stack(List type 활용)
    for i in range(len(progresses)):
        days_left = 100-progresses[i]
        if days_left%speeds[i]==0:
            days_spent = days_left//speeds[i]
        else:
            days_spent = days_left//speeds[i] + 1
    
        days_list.append(days_spent)
        days_spent = 0

    
    # 소요일수가 "오르지 않는 날"의 지속기간 count("주식가격"문제와 동일한 구조)
    cnt, idx = 1, 0
    # stack이되, pop(0)으로 FIFO 구조로 활용
    temp = days_list.pop(0)
    while len(days_list) > idx:
        
        print(f"days_list : {days_list}, temp : {temp}, idx : {idx}, cnt : {cnt}")
        
        if temp >= days_list[idx]:
            cnt += 1
            days_list.pop(0)
        
            if len(days_list)<=1:
                answer.append(cnt)
                cnt = 1  # 이걸 빼먹어서, solution([93, 30, 55], [1, 30, 5]) 케이스에서 [2,1]이 아닌 [2,2]가 나온 것

        else:
            answer.append(cnt)
            cnt = 1
            temp = days_list.pop(0)
    
    return answer
```


```python
solution([93, 30, 55], [1, 30, 5])
```

    days_list : [3, 9], temp : 7, idx : 0, cnt : 1
    days_list : [9], temp : 7, idx : 0, cnt : 1





    [2, 1]




```python
solution([95, 90, 99, 99, 80, 99],[1, 1, 1, 1, 1, 1])
```

    days_list : [10, 1, 1, 20, 1], temp : 5, idx : 0, cnt : 1
    days_list : [1, 1, 20, 1], temp : 10, idx : 0, cnt : 1
    days_list : [1, 20, 1], temp : 10, idx : 0, cnt : 2
    days_list : [20, 1], temp : 10, idx : 0, cnt : 3
    days_list : [1], temp : 20, idx : 0, cnt : 1





    [1, 3, 2]



### 결과

- 11개 case 중 10개 틀림

### 3차)

**디버깅**


```python
def solution(progresses, speeds):
    answer = []
    days_list = []
    
    # 각 task별 소요일수 계산 후, task 우선순위대로 소요일수를 stack(List type 활용)
    for i in range(len(progresses)):
        days_left = 100-progresses[i]
        if days_left%speeds[i]==0:
            days_spent = days_left//speeds[i]
        else:
            days_spent = days_left//speeds[i] + 1
    
        days_list.append(days_spent)
        days_spent = 0

    
    # 소요일수가 "오르지 않는 날"의 지속기간 count("주식가격"문제와 동일한 구조)
    cnt, idx = 1, 0
    # stack이되, pop(0)으로 FIFO 구조로 활용
    temp = days_list.pop(0)
    while len(days_list) > idx:
        
        if temp >= days_list[idx]:
            cnt += 1
            days_list.pop(0)
        
            if len(days_list)<=1:
                answer.append(cnt)
                cnt = 1

        else:
            answer.append(cnt)
            cnt = 1
            temp = days_list.pop(0)
    
    return answer
```


```python
solution([93, 30, 55], [1, 30, 5])
```




    [2, 1]




```python
solution([95, 90, 99, 99, 80, 99],[1, 1, 1, 1, 1, 1])
```




    [1, 3, 2]




```python
solution([93, 30, 55], [1, 1, 5])
```




    [1, 2]




```python
# 이 케이스에서 오답!
# 정답 :  [1,1,1], 출력값 : [1,1]
solution([93, 30, 55], [1, 9, 5])
```




    [1, 1]



**수정**

- days_list가 비면, whlie문을 벗어나 기존의 누적된 cnt를 그대로 append! (answer.append(cnt))
- while문은 "value값이 오르지 않는 날" 의 지속기간을 count하는 역할. 이는 days_list가 빈 컨테이너가 되면(더이상 비교 대상이 없게되면), while문의 역할을 더이상 할 수 없게 된다.
- 따라서, 이 경우, while문을 벗어나 그대로 cnt를 출력하도록 해야 함.
- 위의 디버깅 코드에선, days_list = [] 인 경우, while문 안에 들어갈 수 없어, answer에 아무것도 append되지 않는 문제가 발생했던 것


```python
def solution(progresses, speeds):
    answer = []
    days_list = []
    
    # 각 task별 소요일수 계산 후, task 우선순위대로 소요일수를 stack(List type 활용)
    for i in range(len(progresses)):
        days_left = 100-progresses[i]
        if days_left%speeds[i]==0:
            days_spent = days_left//speeds[i]
        else:
            days_spent = days_left//speeds[i] + 1
    
        days_list.append(days_spent)
        days_spent = 0

    
    # 소요일수가 "오르지 않는 날"의 지속기간 count("주식가격"문제와 동일한 구조)
    cnt, idx = 1, 0
    # stack이되, pop(0)으로 FIFO 구조로 활용
    temp = days_list.pop(0)
    
    ## 수정 1
    while days_list:
        
        if temp >= days_list[idx]:
            cnt += 1
            days_list.pop(0)

        else:
            answer.append(cnt)
            cnt = 1
            temp = days_list.pop(0)
        
    ## 수정 2
    answer.append(cnt)
    
    return answer
```


```python
# 이 케이스에서 오답!
# 정답 :  [1,1,1], 출력값 : [1,1]
solution([93, 30, 55], [1, 9, 5])
```

    days_list : [8, 9], temp : 7, idx : 0, cnt : 1
    days_list : [9], temp : 8, idx : 0, cnt : 1





    [1, 1, 1]




```python
solution([95, 90, 99, 99, 80, 99],[1, 1, 1, 1, 1, 1])
```

    days_list : [10, 1, 1, 20, 1], temp : 5, idx : 0, cnt : 1
    days_list : [1, 1, 20, 1], temp : 10, idx : 0, cnt : 1
    days_list : [1, 20, 1], temp : 10, idx : 0, cnt : 2
    days_list : [20, 1], temp : 10, idx : 0, cnt : 3
    days_list : [1], temp : 20, idx : 0, cnt : 1





    [1, 3, 2]




```python
solution([93, 30, 55], [1, 1, 5])
```

    days_list : [70, 9], temp : 7, idx : 0, cnt : 1
    days_list : [9], temp : 70, idx : 0, cnt : 1





    [1, 2]


