# chapter 04-03
# 시퀀스 형

# 파이썬의 자료구조 형태 : 컨테이너형 / 플랫 또는 가변형 / 불변형
# 자료형 1. 컨테이너(Container) : 서로 다른 자료형을 담을 수 있음 (리스트, 튜플, collections.deque)
# 컨테이너 ex) a = [3, 3.0, 'a', [1,2]]
# 자료형 2. 플랫(Flat) : 한 개의 자료형을 담을 수 있음 (str, bytes, bytearray, array.array, memoryview)

# 자료형 1. 가변 : 리스트, bytearray, array.array, memoryview, collections.deque)
# 자료형 2. 불변 : 튜플, str, bytes

# 해시테이블
# 해시테이블 : key에 value를 저장하는 구조   ex) dict
# key값을 해싱함수 -> 해시 주소 -> key에 대한 value참조

# hash값 확인 (list type은 unhashable)
t1 = (1,2,(3,4,5))
t2 = (1,2,[3,4,5])

print(hash(t1))
# print(hash(t2))  # t2는 unhashable

print('\n')


# Dict Setdefault : tuple -> dictionary 생성 시, 공식 권장사항. 필히 익힐 것.
### source를 dictionary로 변경 시, key가 'k1', 'k2'에 value가 여러개라, key값 중복으로 온전히 dict 생성 못할 수 있음
source = (('k1','val1'),
          ('k1','val2'),
          ('k2','val3'),
          ('k2','val4'),
          ('k2','val5'))

new_dict1 = {}
new_dict2 = {}

# Not use Setdefault
for k, v in source:
    if k in new_dict1:
        new_dict1[k].append(v)
    else:
        new_dict1[k] = [v]
print(new_dict1)

# use Setdefault : key값이 중복(충돌) 시, value 공간 생성 해 문제 해결
# Setdefault : tuple -> dict 시, 표준!
for k, v in source:
    new_dict2.setdefault(k, []).append(v)
print(new_dict2)
