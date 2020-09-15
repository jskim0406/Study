# chapter 03_01
# 파이썬의 핵심 -> sequence, iterator, functions, class
# special method(=magic method)
# 클래스 안에 정의할 수 있는 특별한(Built-in) 메서드

# 자료형은 모두 파이썬 내부적으로 클래스로 구성되어있다.
print(int)
print(float)
print(list)
print()
print()

print(dir(int))
a = 10
print(a.__add__(10))
print(a.__divmod__(6))
print(a.__dir__())
print(a.__eq__(10))
print(a.__bool__())
print(a.__doc__)
print(a.__mul__(10))

print()
print()

# 클래스 예제 1
# 매직메서드 오버라이딩으로 기본 사칙연산 등 기호의 역할/의미 커스터마이징
class Fruit:
    def __init__(self,name,price):
        self._name = name
        self._price = price

    # object + object 하면, 알아서 각 object의 어트리뷰트 중 price 더해주게 만들고 싶음
    def __add__(self,x):
        return self._price + x._price

    def __le__(self,x):
        if self._price <= x._price:
            return True
        else:
            return False

    def __ge__(self,x):
        if self._price >= x._price:
            return True
        else:
            return False

    def __eq__(self,x):
        if self._price == x._price:
            return True
        else:
            return False

    def __mul__(self,x):
        return self._price * x._price

f1 = Fruit('orange',8000)
f2 = Fruit('orange',2000)

print(f1+f2)
print(f1<=f2)
print(f1>=f2)
print(f1==f2)
print(f1*f2)






print()
print()
