def simple_interest(p, r, t):
    interest = p * r * t
    return interest / 100


name = input("Enter Your name: ")
print(f"Hey {name}, I am your simple interest Calculator")
p = eval(input("Enter Principal: "))
r = eval(input("Enter Rate: "))
t = eval(input("Enter Time: "))

answer = simple_interest(p, r, t)
print(f"Hello {name} your simple interest is {answer}")
