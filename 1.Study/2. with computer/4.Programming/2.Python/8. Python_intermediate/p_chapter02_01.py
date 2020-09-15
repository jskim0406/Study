# chapter 02-01
# 객체지향 프로그래밍(OOP) (<-> 절차지향) 장점 : 코드 재사용, 코드 중복 방지, 유지 보수 쉬움, 대형 프로젝트 관리 용이
# 규모가 큰 프로젝트 수행 시, 과거에는 함수 중심으로 코딩됨(함수에서 함수 호출하며 복잡해짐) -> 데이터가 방대해질 수록 개선 어려움 (구조 복잡)
# 클래스 중심 -> 객체로 관리

# 일반적인 코딩
# 차량 1
car_company1 = 'Ferrari'
car_detail1 = [
{'color' : 'white'},
{'horse_power' : 400}
]

# 차량2
car_company2 = 'BMW'
car_detail2 = [
{'color' : 'black'},
{'horse_power' : 270}
]

# 차량3
car_company3 = 'Audi'
car_detail3 = [
{'color' : 'orange'},
{'horse_power' : 350}
]


# 리스트 구조
# 관리하기 불편, 인덱스로 접근해야 함 (딕셔너리는 key값으로 조회가 가능)
car_company_list = ['Ferrari','BMW','Audi']
car_detail_list = [
{'color' : 'white', 'horse_power' : 400},
{'color' : 'black', 'horse_power' : 270},
{'color' : 'orange', 'horse_power' : 350}
]

# del car_company_list[1]
# del car_detail_list[1]
# 
# print(car_company_list, car_detail_list)


# 딕셔너리 구조
car_dicts = [
{'car_company' : 'Ferrari', 'car_detail' : {'color' : 'white', 'horse_power' : 400}},
{'car_company' : 'BMW', 'car_detail' : {'color' : 'black', 'horse_power' : 270}},
{'car_company' : 'Audi', 'car_detail' : {'color' : 'orange', 'horse_power' : 350}}
]

print(car_dicts[0]['car_company'],car_dicts[0]['car_detail'])
print()
print()


# 클래스 구조
# 구조 설계 후 : 재사용성 증가, 코드 반복 최소화, 메소드 활용

class Car():
    def __init__(self,company,detail):
        self._company = company
        self._detail = detail
        
    # print(class object) 시 리턴 값 ex) print(car1)
    def __str__(self):
        return 'str : {} - {}'.format(self._company,self._detail)
        
    # class object 호출 시 리턴 값 ex) car1 
    def __repr__(self):
        return 'repr : {} - {}'.format(self._company,self._detail)
        
        
car1 = Car('Ferrari',{'color' : 'white','horse_power' : 400})
car2 = Car('BMW',{'color' : 'black','horse_power' : 270})
car3 = Car('Audi',{'color' : 'orange','horse_power' : 350})

# 객체.__dict__ : 객체의 attribute, value값 확인 가능
print(car1.__dict__)
print(car2.__dict__)
print(car3.__dict__)
print()
print()

# 객체 내 메타정보 확인 가능(매직 메소드)
print(dir(car1))
print()
print()

car_list = []
car_list.append(car1)
car_list.append(car2)
car_list.append(car3)

# __repr__로 설정된 값들이 객체마다 출력될 것
print(car_list)
print()
print()

for x in car_list:
    print(x)
    
print()
print()




