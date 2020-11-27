# Chapter 06-03
# 병행성(Concurrency) : 한 컴퓨터(cpu)가 여러 일을 동시에 수행 -> 단일 프로그램 안에서 여러 일을 쉽게 해결
# 병렬성(parallelism) : 여러 컴퓨터(cpu)가 여러 일을 동시에 수행 -> 속도가 높음

# Generator ex3) (중요 함수)
### count, takewhile, filterfalse, accumulate, chain, product, product, groupby
import itertools

gen1 = itertools.count(1, 3)

print(next(gen1))
print(next(gen1))
print(next(gen1))
print(next(gen1))
# ... 무한

print()

# 조건
gen2 = itertools.takewhile(lambda n : n < 10, itertools.count(1, 2.5))

for v in gen2:
    print(v)


print()

# 필터 반대
gen3 = itertools.filterfalse(lambda n : n < 3, [1,2,3,4,5])

for v in gen3:
    print(v)


print()

# 누적 합계
gen4 = itertools.accumulate([x for x in range(1, 11)])

for v in gen4:
    print(v)

print()

# 연결1 : 서로 다른 iterable 을 묶어주는 것
gen5 = itertools.chain('ABCDE', range(1,11,2))

print(list(gen5))

# 연결2

gen6 = itertools.chain(enumerate('ABCDE'))

print(list(gen6))

# 개별 : 개별로 분리해서 tuple화 ==> [('A',),('B',), ...]
gen7 = itertools.product('ABCDE')

print(list(gen7))

# 연산(경우의 수) : 개별로 분리해서 조합을 tuple화 ==> [('A','A'),('A','B'),('A','C'), ...]
gen8 = itertools.product('ABCDE', repeat=2)

print(list(gen8))

# 그룹화
gen9 = itertools.groupby('AAABBCCCCDDEEE')

# print(list(gen9))

for chr, group in gen9:
    print(chr, ' : ', list(group))

print()
