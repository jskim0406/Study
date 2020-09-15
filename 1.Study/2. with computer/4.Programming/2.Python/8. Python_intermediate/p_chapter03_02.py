# 클래스 예제 2
# 벡터 계산 클래스 생성
# ex) (5,2) + (3,4) = (8,6) 이 되도록
# (10,3)*5 = (50,15)
# max((5,10)) = 10
class Vector():

    def __init__(self, *args):
        '''Create vector. ex) v = Vector(1,2,3,4,5) ==>> v = (1,2,3,4,5)'''
        self.v = args
        print("vector created : {}, {}".format(self.v, type(self.v)))

    def __repr__(self):
        '''Return vector information'''
        return self.v

    def __getitem__(self,key):
        return self.v[key]

    def __len__(self):
        return len(self.v)

    def __add__(self, other):
        '''Return element-wise operation of sum'''
        if type(other) != int:
            temp = ()
            for i  in range(len(self.v)):
                temp += (self.v[i] + other[i],)
            return temp

        else:
            temp = ()
            for i  in range(len(self.v)):
                temp += (self.v[i] + other,)
            return temp

    def __mul__(self, other):
        '''Return Hadamard Product. element-wise operation'''
        if type(other) != int:
            temp = ()
            for i  in range(len(self.v)):
                temp += (self.v[i] * other[i],)
            return temp

        else:
            temp = ()
            for i  in range(len(self.v)):
                temp += (self.v[i] * other,)
            return temp


v1 = Vector(1,2,3,4,5)
v2 = Vector(10,20,30,40,50)

print()
print(v1.__add__.__doc__)
print()
print(v1+v2, v1*v2, sep='\n')
print()
print(v1+3, v1*3, sep='\n')
print()
