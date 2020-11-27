# Chapter 05-03
# 일급 함수 (일급 객체)
# 클로저 기초
# 클로저 : 외부에서 호출된 함수의 변수값, 상태(레퍼런스) 복사 후 저장 -> 이후에 접근(엑세스) 가능하게끔 함

def closure_ex1():
    series = []    # series >>> free variable : 상태를 복사 후 저장하는 변수 (inner function을 free하게 관통함)

    def averager(v):
        series.append(v)
        print(f"inner : {series}, {sum(series)/len(series)}")
        return sum(series) / len(series)

    return averager


ex1 = closure_ex1()
ex1(10)
ex1(30)
ex1(50)
ex1(70)

print('\n')

print(dir(ex1))
print(dir(ex1.__closure__))
print(dir(ex1.__code__))
print()
print(ex1.__code__.co_freevars)
print(ex1.__closure__[0].cell_contents)

print('\n')
# ### 잘못된 클로저 사용
# ### cnt, total이 free variable 역할을 하지 못함.
# def closure_ex2():
#
#     cnt, total = 0, 0
#     def averager(v):
#         cnt += 1
#         total += v
#
#         return total / cnt
#
#     return averager
#
# ex2 = closure_ex2()
# ex2(10)    # UnboundLocalError: local variable 'cnt' referenced before assignment

print('\n')
##### 디버깅
def closure_ex3():

    cnt, total = 0, 0
    def averager(v):
        nonlocal cnt, total
        cnt += 1
        total += v

        return total / cnt

    return averager

ex3 = closure_ex3()
print(ex3(10))
print(ex3(30))
print(ex3(50))
