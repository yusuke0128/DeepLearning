class Test(object):
    c_var = "Hello"

    def __init__(self):
        self.i_var = "World"

    def __str__(self):
        return Test.c_var + " " + self.i_var

    def examine1(self):
        self.c_var = "Hey"
        return self

    def examine2(self):
        Test.c_var = "Hey"
        return self

t = Test()
print(t.examine1())
print(t.examine2())
print(Test())
