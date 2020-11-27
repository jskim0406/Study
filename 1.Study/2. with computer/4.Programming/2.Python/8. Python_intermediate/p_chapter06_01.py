# Chapter 06-01
# 병행성(Concurrency)
# generator : function that returns an object(iterator) which we can iterate over (one value at a time)
# iterator : iterable object

# 파이썬의 iterable(반복가능한) type
# collections, string, list, dict, set, tuple, unpacking, *args ... : iterable

# iterable type이 반복 가능한 이유, 그 과정
t = 'ABCDEFGHIJK'

# ex) for 반복문의 실행 구조
for c in t:
    print("for iteration >> ", c)

print()

### for 반복문의 구조는 사실, iterable 객체를 iter(t)를 통해 iterable 객체를 iterator로 만든 후 next function으로 하나씩 꺼내는 과정
### iterable object의 element 꺼내기 : iter( ) -> next () 구조
w = iter(t)

while True:
    try:
        print(next(w))
    except StopIteration:
        break

print()


# 그렇다면, iterable type인지 어떻게 확인하는 가?
# answer : magic method 중 ' __iter__ ' 가 있다면 iterable object 이다.
### 방법 1. dir(object)로 magic method 목록 조회
print(dir(w))

### 방법 2. hasattr(object, '__iter__') 를 통해 __iter__ 를 갖고 있는 지 조회
print(hasattr(w,'__iter__'))

### 방법 3. collections 모듈의 'abc'(abstract class) 조회 기능을 활용해 Iterable class를 갖는 지 조회
from collections import abc
print(isinstance(w, abc.Iterable))

print('\n')



# iterator의 next 함수
# class 자체를 iterable 하게 만든 예시
class WordSplitter:
    def __init__(self, text):
        self._idx = 0
        self._text = text.split(' ')

    def __next__(self):
        # print('Called __next__')
        try:
            word = self._text[self._idx]    # self._idx를 따로 생성해, 위치정보(idx)를 기억 -> generator의 yield에서 대체됨
        except IndexError:
            raise StopIteration('Stopped Iteration.')
        self._idx += 1
        return word

    def __repr__(self):
        return 'WordSplit(%s)' % (self._text)


wi = WordSplitter('Do today what you could do tomorrow')

print(wi)
print(next(wi))
print(next(wi))
print(next(wi))
print(next(wi))
print(next(wi))
print(next(wi))
print(next(wi))
# print(next(wi))

print()
print()

# Generator 패턴
# 1.지능형 리스트, 딕셔너리, 집합 -> 데이터 양 증가 후 메모리 사용량 증가 -> 제네레이터 사용 권장
# 2.단위 실행 가능한 코루틴(Coroutine) 구현과 연동
# 3.작은 메모리 조각 사용

class WordSplitGenerator:
    def __init__(self, text):
        self._text = text.split(' ')

    def __iter__(self):
        # print('Called __iter__')
        for word in self._text:
           yield word # 제네레이터  # yield의 역할 : idx를 기억해 'self._idx += 1'을 할 필요 없게 한다.
        return

    def __repr__(self):
        return 'WordSplit(%s)' % (self._text)


wg = WordSplitGenerator('Do today what you could do tomorrow')

wt = iter(wg)

print(wt)
print(next(wt))
print(next(wt))
print(next(wt))
print(next(wt))
print(next(wt))
print(next(wt))
print(next(wt))
# print(next(wt))

print()
print()



# 실습 연습 : generator를 class로 구현 실습 연습 
class gen_test:
    def __init__(self,data):
        self._data = data

    def __iter__(self):
        for char in self._data:
            yield char

te = gen_test("hello world hi world")
tt = iter(te)

print(next(tt))
print(next(tt))
print(next(tt))
