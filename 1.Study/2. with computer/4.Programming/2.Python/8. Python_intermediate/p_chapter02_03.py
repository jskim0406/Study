# chapter 02-01
# 객체지향 프로그래밍(OOP) (<-> 절차지향) 장점 : 코드 재사용, 코드 중복 방지, 유지 보수 쉬움, 대형 프로젝트 관리 용이
# 규모가 큰 프로젝트 수행 시, 과거에는 함수 중심으로 코딩됨(함수에서 함수 호출하며 복잡해짐) -> 데이터가 방대해질 수록 개선 어려움
# 클래스 중심 -> 객체로 관리

class Car():
    """
    Car class
    Author : Kim
    Date : 2020.08.31
    Description : Class, Static, Instance method
    """

    # 클래스 변수 (모든 인스턴스가 공유. 각 객체의 인스턴스로 선언 안해도(__init__에서 안해줘도) 쓸 수 있다.)
    # ex) # A 처럼 해놓으면, 서로 다른 객체가 생성될 때마다, count += 1 된다. (서로 다른 객체가 count 인스턴스 공유함을 의미)
    # 클래스 변수와 동일한 네이밍으로 인스턴스 변수 생성 가능 -> 이때, 인스턴스 변수를 먼저 접근한다.
    price_per_raise = 1.0

    # 인스턴스 변수 네이밍 : _company
    # 클래스 변수 네이밍 : company

    def __init__(self,company,detail):
        self._company = company
        self._detail = detail

    # print(class object) 시 리턴 값 ex) print(car1)
    def __str__(self):
        return 'str : {} - {}'.format(self._company,self._detail)

    # class object 호출 시 리턴 값 ex) car1
    def __repr__(self):
        return 'repr : {} - {}'.format(self._company,self._detail)

    # instance method : 우리가 지금까지 사용해왔던 method
    # self : Class 객체의 고유한 속성값 사용한다는 의미
    def detail_info(self):
        print('Current ID : {}'.format(id(self)))
        print('Car detail info : {} {}'.format(self._company, self._detail.get('price')))

    # Instance method
    def get_price(self):
        return 'before car price -> company : {}, price : {}'.format(self._company,self._detail['price'])

    # Instance method
    def get_price_calc(self):
        return 'after car price -> company : {}, price : {}'.format(self._company,self._detail['price']*Car.price_per_raise)

    # class method
    # class 변수('price_per_raise'같은) 수정 등 시 사용
    @classmethod
    def raise_price(cls, per):
        if per <= 1:
            print('please enter 1 or more')
            return
        cls.price_per_raise = per
        print('succeed!, price increased.')

    # Static method
    # 클래스 안에서 공통적인 기능을 수행하는 함수
    # self, cls인자 안받고, Car.fn, car1.fn으로도 호출 가능한 유연한 함수
    @staticmethod
    def is_bmw(abcde):
        if abcde._company == 'BMW':
            return "ok. this car is {}.".format(abcde._company)
        return "sorry. this car is not bmw."


# self의 의미 : 각 object를 지정하기 위한 지정
car1 = Car('Ferrari',{'color' : 'white','horse_power' : 400,'price':8000})
car2 = Car('BMW',{'color' : 'black','horse_power' : 270,'price':5000})

# 전체정보
car1.detail_info()
car2.detail_info()

# 가격정보 (직접 접근)
print(car1._detail['price'])
print(car2._detail['price'])

print()
print()

# 가격정보 (인상 전)
print(car1.get_price())
print(car2.get_price())

print()
print()

# 가격 인상 후(클래스 메소드 미사용)
Car.price_per_raise = 1.4
print(car1.get_price_calc())
print(car2.get_price_calc())

print()
print()

# class method 활용해 가격 인상
# 클래스 변수 변경 시, class method 활용!
Car.raise_price(1.6)
print(car1.get_price_calc())
print(car2.get_price_calc())

print()
print()


# instance method : self
# class method : cls
# static method : instance(self)나 class(cls)를 모두 받으며(특정할 필요 없음), 클래스 내 모든 어트리뷰트에 연결가능
# instance로 호출 (@staticmethod)
print(car1.is_bmw(car1))
print(car1.is_bmw(car2))
# class로 호출 (@staticmethod)
print(Car.is_bmw(car1))
