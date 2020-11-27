# Chapter 05-01
# First-class : 일급함수 -> '함수를 객체로 취급!'

# 파이썬 함수 특징
# 1. 함수 실행 시, 런타임 초기화
# 2. 함수를 변수에 할당 가능함
# 3. 함수를 인수로 전달 가능
# 4. 함수의 결과를 리턴 가능

def factorial(n):
    '''factorial operation : n -> int'''
    if n<2:
        return 1
    return n * factorial(n-1)

# 함수를 객체 취급하는 가? 증명
print(type(factorial),'  ',factorial(5))
print('\n')

# 함수를 변수에 할당 가능? 증명
var_func = factorial
print(var_func)
print(var_func(5))
print(list(map(var_func,range(1,10+1))))
print('\n')

# 함수를 인수로 전달 및 함수로 결과 반환 가능? 증명
# 이를 '고위 함수' 라고 함 (higher-order function)
# map, filter, reduce
print([var_func(i) for i in range(1,10+1) if i%2])
print(list(map(var_func, filter(lambda x: x%2, range(1,10+1)))))     # map()의 인수로서 var_func이 전달됨.
print('\n')

# reduce
from functools import reduce
from operator import add
print(sum(range(1,10+1)))   # reduce보다 이게 더 효율적인 코드
print(reduce(add, range(1,10+1)))

# 익명함수(lambda) : 사용 시, 가급적 주석 표기
# add를 lambda로 대체
print(reduce(lambda x,y: x+y, range(1,10+1)))
print('\n')

# Callable : 호출 연산자 -> 메서드 형태로 호출 가능한지 확인
print(dir(factorial))     # __call__ 이 있으므로, callable 함을 확인
print(callable(str), callable(var_func), callable(factorial), callable(3.14))

# partial 함수 사용법 : 인수를 고정 -> 콜백 함수에 주료 사용
from functools import partial
from operator import mul

### 인수 고정 5
five = partial(mul, 5)
print(five(10))   # 5 * 10 이지만, 5라는 인수는 고정해놓았기 때문에(partial 함수) 5를 따로 함수의 인수로 넣을 필요 없음
### 인수 '추가' 고정 6
six = partial(five, 6)    # five라는 함수의 인수로 6을 새로 고정 -> 이로서, five의 인수로 5,6이 고정됨
print(six())
# print(six(7))    # five함수의 필요 인수 2개가 모두 고정된 상태이기 때문에, 추가로 인수를 넣으면 typeerror 발생
