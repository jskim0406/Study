# chapter 04-01
# 시퀀스 형

# 파이썬의 자료구조 형태 : 컨테이너형 / 플랫 또는 가변형 / 불변형
# 자료형 1. 컨테이너(Container) : 서로 다른 자료형을 담을 수 있음 (리스트, 튜플, collections.deque)
# 컨테이너 ex) a = [3, 3.0, 'a', [1,2]]
# 자료형 2. 플랫(Flat) : 한 개의 자료형을 담을 수 있음 (str, bytes, bytearray, array.array, memoryview)

# 자료형 1. 가변 : 리스트, bytearray, array.array, memoryview, collections.deque)
# 자료형 2. 불변 : 튜플, str, bytes


# 1. 리스트 및 튜플 고급
# 지능형 리스트(comprehension lists)
# ord() : 문자가 unicode 상의 할당된 숫자를 반환하는 함수
# chr() : 숫자를 unicode 상 할당된 문자로 반환하는 함수 (ord()의 반대기능)
### 유니코드 리스트 실습
chars = '+_)(*&^%$#@!)'
chars_2 = 'abcdefghijklmnop'
uni_list = [ord(s) for s in chars]
uni_list_ = [ord(s) for s in chars_2]
print(uni_list)
print(uni_list_)

### comprehension list + map/filter (unicode로 40이상 인 경우만 출력)
uni_list2 = [ord(s) for s in chars if ord(s)>40]
uni_list3 = list(filter(lambda x: x>40, map(ord, chars)))
print(uni_list2)
print(uni_list3)

print()
print()

### chr()활용
print(uni_list)
print(uni_list2)
print(uni_list3)
print()
print([chr(num) for num in uni_list])
print([chr(num) for num in uni_list2])
print([chr(num) for num in uni_list3])
print()


# 2. Generator
# Generator : 규칙만 유지. 한 번에 한 개의 항목을 생성(미리 모든 경우를 메모리에 저장해놓지 않음)
# Generator : 적은 메모리로, iterator를 만들어 낼 수 있음
### 참고 1. iterator : 반복해서 조회 가능한 객체를 생성(자신의 멤버를 한 번에 하나씩 리턴할 수 있는 객체를 생성)
### iterator인지 어떻게 판단?
##### a = [1,2,3,4], dir(a)로 나오는 magic-method 중 '__iter__'가 있다면 iterable 객체

# generator 실습 1 : [] 가 아닌 () 로 바꿔주면 끝
list_ = [ord(s) for s in chars]
print("list : ", list_)
gen_ = (ord(s) for s in chars)
print("generator : ", gen_)

print(type(gen_))
print('1st_entry : ', next(gen_))
print('2nd_entry : ', next(gen_))

for num in gen_:
    print(num)

print()
print()


# generator 실습 2 : class-student_num 조합 generator 생성 실습
comb_list = [f'{rank}' + str(num)
        for rank in 'A B C D'.split(' ')
            for num in range(1,21)]
print(comb_list)

comb_gen = (f'{rank}' + str(num)
        for rank in 'A B C D'.split(' ')
            for num in range(1,21))
print(comb_gen)

for temp in comb_gen:
    print(temp)


# 리스트 생성 주의
### 생성 방법에 따라, id가 같은 리스트를 복사하는 방법이 있어 주의해야함
temp = [['~']*3 for _ in range(4)]
print(temp)

temp2 = [['~']*3]*4
print(temp2)

### 과연 temp1 == temp2 인가? NO! 매우 중요한 차이가 있다.
### 아래처럼 재할당 시, temp2는 [0]요소 외 모든 element가 함께 바뀌는 것을 알 수 있다.
### Why? temp2는 []*4 은 []를 4개 '복사'하라는 의미이다.
###      하지만, temp1의 list-comprehension은 리스트 '생성' 방법이기에, 다른 리스트가 새로 생성되는 것!
temp[0][0] = 'X'
temp2[0][0] = 'X'
print(temp)
print(temp2)
print()
print()

### 증명 (id값이 서로 다름을 확인 가능)
### 역시나, []*4 는 동일한 []를 '복제'하는 것이고, list-comprehension은 []를 깊은 복사로 새로이 '생성'하는 것
print([id(i) for i in temp])
print([id(i) for i in temp2])
