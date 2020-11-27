# Chapter 05-02
# 일급 함수 (일급 객체)
# 클로저 기초

# 파이썬의 변수 범위(scope)

# ex1
b = 10
def temp_func1(a):
    print("ex1")
    print(a)
    print(b)

temp_func1(20)
print('\n')

# ex2
b = 10
def temp_func2(a):
    print("ex2")
    b = 5
    print(a)
    print(b)

temp_func2(20)
print('\n')

# ex3
# b = 10
# def temp_func3(a):
    # print("ex3")
#     b = 5
#     global b
#     print(a)
#     print(b)

# temp_func3(20)    -> SyntaxError: name 'b' is assigned to before global declaration

# ex4
b = 10
def temp_func4(a):
    print("ex4")
    global b
    print(a)
    print(b)

temp_func4(20)
print('\n')

# ex5
b = 10
def temp_func5(a):
    print("ex5")
    global b
    print(a)
    print(b)
    b = 100    # temp_func5 의 scope안에서 b는 global variable로 지정되었기 때문에, b = 100으로 인해 global variable b가 100으로 재할당 됨

temp_func5(20)
print(b)

print('\n')



# Closure : "function object that remembers values in enclosing scopes"
# closure 사용 이유 : temp_func5 내부 scope 변수 'a'를 scope밖에서도 기억 + 사용하기 위해
# closure : "remember"

### class 이용한 closure 개념 구현 : "기억"
class Averager():
    def __init__(self):
        self._list = []

    def __call__(self, v):  # class를 call 시, 작동
        self._list.append(v)
        print(f"inner : {self._list} / {len(self._list)}")
        print(sum(self._list) / len(self._list))

ave = Averager()
ave(10)
ave(20)
ave(30)
ave(40)
ave(50)    # self._list 라는 공간 안에서 앞에서 입력된 값들을 지속해서 "기억"하는 역할을 함 == closure 의 역할
