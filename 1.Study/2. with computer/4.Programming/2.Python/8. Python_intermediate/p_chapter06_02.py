# Chapter 06-01
# 병행성(Concurrency) : 한 컴퓨터(cpu)가 여러 일을 동시에 수행 -> 단일 프로그램 안에서 여러 일을 쉽게 해결
# 병렬성(parallelism) : 여러 컴퓨터(cpu)가 여러 일을 동시에 수행 -> 속도가 높음

# Generator ex1)
def generator_ex1():
    print("  start")
    yield 'A Point'
    print("  continue")
    yield 'B Point'
    print("  end")
    yield "C Point"


print(hasattr(generator_ex1(), '__iter__'))  # True. why? 'yield'가 function을 iterable하게 만들었기 때문!

iter_ = generator_ex1()

print(next(iter_))
print(next(iter_))
print(next(iter_))
print('\n')

iter_2 = iter(generator_ex1())  # 'generator_ex1()' 자체가 iterable하기에, iter( )로 감싸줄 필요는 없긴 함.

print(next(iter_2))
print(next(iter_2))
print(next(iter_2))

print('\n')


# Generator ex2)
temp = [x*2 for x in generator_ex1()]    # list_comprehension : 모든 결과를 한번에 list에 저장
temp2 = (x*2 for x in generator_ex1())    # generator obejct로서, 한 번에 하나씩만 꺼냄

### temp는 list_comprehension. generator_ex1() 함수 안의 'yield'가 return 역할도 하기에, temp 에 결과 값들이 저장되는 것
print('\n')
print(temp)

### temp2는 generator 이기에, 출력 시, next 함수 활용
print(temp2)
next(temp2)

print('\n')

for v in temp:
    print(v)

print('\n')

for v in temp2:
    print(v)     # line 43에서 한번 yield를 했기에, for문은 첫 yield 다음인 'continue'부터 시작됨! (yield가 위치를 기억)
