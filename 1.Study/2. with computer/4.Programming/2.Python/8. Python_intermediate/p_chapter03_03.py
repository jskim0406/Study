# 객체 -> 파이썬의 데이터를 추상화
# 모든 객체 -> id, type을 갖는다 -> 그리고 value를 갖는다

# named tuple : tuple인데, dictionary 의 특징을 갖음 => key와 index로 접근 가능
from collections import namedtuple

# named tuple 선언
Point = namedtuple('Point','x y z')

pt1 = Point(1.0,2.0,3.0)
pt2 = Point(10.0,2.0,3.0)

# key와 index로 접근 가능
print(pt1, pt2, sep='\n')
print()
print(pt1[0])
print(pt1.x)
print()
print(pt2[1],pt2.y,sep='\n')

# namedtuple 선언 방법
Point1 = namedtuple('Point', 'x y')
Point2 = namedtuple('Point', 'x, y')
Point3 = namedtuple('Point', ['x', 'y'])
Point4 = namedtuple('Point', 'x y x class', rename=True) # 예약어, 중복된 값(x) 등을 선언 시 사용하려면, rename=True로 설정해야함

print(Point1, Point2, Point3, Point4, sep='\n')

# 객체 생성 방법
p1 = Point1(1,2)
p2 = Point2(x=1, y=2)
p3 = Point3(1, y=2)
p4 = Point4(1,2,3,4)

print(p1,p2,p3,p4,sep='\n')
print()
print()

# Dict to unpacking
temp_dict = {'a':75, 'b':30}
temp_dict2 = {'x':75, 'y':30}
print(*temp_dict)  # dict의 key를 unpacking
print()
print()

# Dict to namedtuple with unpacking
Point5 = namedtuple('Point','x y')
p5 = Point5(*temp_dict) # Point5의 x = key1, y = key2
p6 = Point5(*temp_dict2) # Point5의 x = key1, y = key2
p7 = Point5(**temp_dict2) # Point5의 x = key1의 value, y = key2의 value
print(p5,p6,p7,sep='\n')
print()

# 사용
print(p7.x + p1[0])
print(p7.x + p1.x)
print()

# unpacking
x, y = p2
print(x,y)
print()

# namedtuple의 method ( _make, _fields, _asdict() )
# _make : 새로운 namedtuple 객체 생성 (list, tuple을 받아)
temp = [100,101]
p100 = Point1._make(temp)
print(p100)
print()
# p101 = Point1(temp) # error 발생 (require argument y)
# print(p101)
temp_tuple = (100,101)
p101 = Point1._make(temp_tuple)
print(p101)
print()
# p102 = Point2(temp_tuple) # error 발생 (require argument y)
# print(p102)

# _fields : field name 확인 (x y 등을 반환)
print(p1._fields, p2._fields)
print()

# _asdict() : OrderedDict 반환
print(p1)
print(p1._asdict())
print(type(p1._asdict()))
print(p7._asdict())
print(type(p7._asdict()))
