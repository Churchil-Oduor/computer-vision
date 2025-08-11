import random

name = input("Enter your name: ")
r = random.randint(1, 10)
ask = eval(input("Guess a number between 1 to 10: "))

if ask != r:
    print("Woooops {}! You got it wrong! the number is {}".format(name, r))
else:
    print("Hurray {}! you guessed it right".format(name))
