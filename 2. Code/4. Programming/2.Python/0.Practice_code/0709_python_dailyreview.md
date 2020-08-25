# 1. 보다 빠른 입력 받기

문제 : https://www.acmicpc.net/problem/15552

```python

# 여러줄 입력받는 것에서 어려움을 느낌 ( 입력속도 : input < sys.stdin.readline() )

# input
a = input()

# sys.stdin.readline() <= 쥬피터노트북에선 실행안 됨. 파이참에서 해야함
a = sys.stdin.readline() 

```

### 1-1. map 함수 사용법

```python

# iterable container로 감싸주기 (list, tuple, set 등)

a = list( map(function, iterable 대상) )

```


```python
a = ['1','2','3']
a = map(int,a)
print(a)
```

    <map object at 0x1a23631ed0>



```python
a = list(map(int,a))
print(a)
```

    [1, 2, 3]



```python
b = tuple(map(int,a))
print(b)
```

    (1, 2, 3)



```python
c = set(map(int,a))
print(c)
```

    {1, 2, 3}


### 1-2. str.rstrip() 

- str.strip() = 문자열의 양 끝 제거
    - () : 공백제거
    - ('<') : '<'제거
- str.rstrip() = 문자열의 오른쪽 끝 제거
- str.lstrip() = 문자열의 왼쪽 끝 제거

### 1-3. str.split()

- str.split() : 문자열을 공백 기준 split
- str.split('<') : 문자열을 < 기준으로 split


```python

```
