# chapter 02-01
# 객체지향 프로그래밍(OOP) (<-> 절차지향) 장점 : 코드 재사용, 코드 중복 방지, 유지 보수 쉬움, 대형 프로젝트 관리 용이
# 규모가 큰 프로젝트 수행 시, 과거에는 함수 중심으로 코딩됨(함수에서 함수 호출하며 복잡해짐) -> 데이터가 방대해질 수록 개선 어려움 (구조 복잡)
# 클래스 중심 -> 객체로 관리
class Car():
    """
    Car class
    Author : Kim
    Date : 2020.08.27
    """

    # 클래스 변수 (모든 인스턴스가 공유. 각 객체의 인스턴스로 선언 안해도(__init__에서 안해줘도) 쓸 수 있다.)
    # ex) # A 처럼 해놓으면, 서로 다른 객체가 생성될 때마다, count += 1 된다. (서로 다른 객체가 count 인스턴스 공유함을 의미)
    # 클래스 변수와 동일한 네이밍으로 인스턴스 변수 생성 가능 -> 이때, 인스턴스 변수를 먼저 접근한다.
    car_count = 0

    # 인스턴스 변수 네이밍 : _company
    # 클래스 변수 네이밍 : company

    def __init__(self,company,detail):
        self._company = company
        self._detail = detail
        Car.car_count += 1  # A

    # print(class object) 시 리턴 값 ex) print(car1)
    def __str__(self):
        return 'str : {} - {}'.format(self._company,self._detail)

    # class object 호출 시 리턴 값 ex) car1
    def __repr__(self):
        return 'repr : {} - {}'.format(self._company,self._detail)

    def detail_info(self):
        print('Current ID : {}'.format(id(self)))
        print('Car detail info : {} {}'.format(self._company, self._detail.get('price')))


# self의 의미 : 각 object를 지정하기 위한 지정
car1 = Car('Ferrari',{'color' : 'white','horse_power' : 400,'price':8000})
car2 = Car('BMW',{'color' : 'black','horse_power' : 270,'price':5000})
car3 = Car('Audi',{'color' : 'orange','horse_power' : 350,'price':6000})

# ID 확인 : 다 다르다. 클래스는 같지만, 객체는 다르다.(self는 각 개체 instance를 지정함)
print(id(car1))
print(id(car2))
print(id(car3))
print()
print()

# dir & __dict__ 확인
# dir( ) : 모든 attribute 출력 : 매직메소드, 어트리뷰트 포함
print(dir(car1))
print()
print()

# __dict__ : 어트리뷰트와 값 출력
print(car1.__dict__)
print()
print()

print(Car.__doc__)
print()
print()

car1.detail_info()
print()
print()

# 비교
print(car1.__class__)
print(car2.__class__)
print()
print()

# 에러
# Car.detail_info() -> self argument가 없다며 에러 발생
Car.detail_info(car1)
print()
print()

# 클래스 변수 : 클래스의 객체가 갖는 변수(def __init__)과 다르다
# 공유 확인
print(car1.car_count)
print()
print()
