# chapter 04-02
# 시퀀스 형

# 파이썬의 자료구조 형태 : 컨테이너형 / 플랫 또는 가변형 / 불변형
# 자료형 1. 컨테이너(Container) : 서로 다른 자료형을 담을 수 있음 (리스트, 튜플, collections.deque)
# 컨테이너 ex) a = [3, 3.0, 'a', [1,2]]
# 자료형 2. 플랫(Flat) : 한 개의 자료형을 담을 수 있음 (str, bytes, bytearray, array.array, memoryview)

# 자료형 1. 가변 : 리스트, bytearray, array.array, memoryview, collections.deque)
# 자료형 2. 불변 : 튜플, str, bytes


# 1. 리스트 및 튜플 고급
### 1) tuple - unpacking
print(divmod(100,9))
print(divmod(*(100,9)))
print(*divmod(100,9)) # tuple unpacking

x,y,*z = range(10)
print(x,y,z)
x,y,*z = 1,2
print(x,y,z)
x,y,*z = 1,2,3,4,5,6
print(x,y,z)

print()
print()

### 2) mutable vs immutable
### ex) list vs tuple => 둘 다 container지만, 가변 vs 불면
### immutable : reassignment 불가

list_ = [1,2,3]
tuple_ = (1,2,3)
print(list_, tuple_)
print('list_ id : ',id(list_), '/ tuple_ id : ',id(tuple_))

list_ = list_*2
tuple_ = tuple_*2
print(list_, tuple_)
print('list_ id : ',id(list_), '/ tuple_ id : ',id(tuple_))

# list는 동일한 주소에 새로운 데이터로 재할당이 가능하지만
# tuple은 동일한 주소에 재할당이 불가능. 따라서, tuple은 새로운 공간에 새롭게 *=2 결과를 할당
list_ *= 2
tuple_ *= 2
print(list_, tuple_)
print('list_ id : ',id(list_), '/ tuple_ id : ',id(tuple_))

print('\n')

### 3) sort vs sorted
##### reverse, key = len, key = str.lower(), key = func..

### sorted : 정렬 후 새 객체 반환
### sort : 정렬 in-place
f_list = ['oragne', 'mango', 'papaya', 'strawberry']

print('sorted : ', sorted(f_list), '/ original : ', f_list)
print('sorted : ', sorted(f_list, reverse=True), '/ original : ', f_list)
print('sorted : ', sorted(f_list, key=len), '/ original : ', f_list)
print('sorted : ', sorted(f_list, key=lambda x: x[-1]), '/ original : ', f_list)

print('sort : ', f_list.sort(), '/ original(in-place됨) : ', f_list)
print('sort : ', f_list.sort(reverse=True), '/ original(in-place됨) : ', f_list)
print('sort : ', f_list.sort(key=len), '/ original(in-place됨) : ', f_list)
print('sort : ', f_list.sort(key=lambda x: x[-1]), '/ original(in-place됨) : ', f_list)
