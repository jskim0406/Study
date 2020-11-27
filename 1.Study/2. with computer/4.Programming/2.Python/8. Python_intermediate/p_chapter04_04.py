# chapter 04-04
# 시퀀스 형
# 해시테이블 -> 적은 리소스로 많은 데이터를 효율적으로 관리
### Dict -> key 중복 허용 X, Set -> 중복 허용 X

# Dict 및 set 심화
# dict(mutable) / dict.setdefault(immutable)
# set(mutable) / frozenset(immutable) : 중복되지 않으면서 오직 읽기 전용으로만 원할 땐 frozenset 사용


# immutable dict : 읽기 전용(수정 불가) 자료형이 되는 것.

from types import MappingProxyType

# 수정 가능, mutable
d = {'key' : 'value1'}

# read-only, immutable : does not support item assignment
d_frozen = MappingProxyType(d)

print(d, id(d))
print(d_frozen, id(d_frozen))

d['key2'] = 'value2'
print(d)
# d_frozen['key2'] = 'value2'       # TypeError: 'mappingproxy' object does not support item assignment


# Set Type
s1 = {'apple', 'orange', 'banana', 'apple'}
s2 = set(['apple', 'orange', 'banana', 'apple'])
s3 = {3}
s4 = set()
s5 = frozenset(['apple', 'orange', 'banana', 'apple'])

print(s1,s2,s3,s4,s5,sep='\n')
print('\n')

s1.add('melon')
print(s1)
# s5(frozenset) : 추가 불가
# s5.add('melon')      # AttributeError: 'frozenset' object has no attribute 'add'


# 선언 최적화  :  a = set([1]) / a = {1}  중 어느 선언 방식이 가장 효율적일까?   답 : {}
# 파이썬 실행 과정 : 바이트코드 실행 -> 어셈블링 후 파이썬 인터프리터 실
from dis import dis
print('----------------------------------------------------------------------')
print(dis( '{10}' ))
print(dis( 'set([10])' ))       # {} 선언이 set([]) 선언보다 더 간단
print('----------------------------------------------------------------------')
