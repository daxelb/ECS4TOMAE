def func(a, b, c):
    return (a,b,c)

if __name__ == "__main__":
    print(func(**{"b": 1, "a": 2, "c": 3}))
    print(*[1,2,3,4,5,6])