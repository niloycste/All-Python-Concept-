# INPUT
current_year=input("Enter Year")
years=2023 -int(current_year)
print(years)

weight=float(input("Enter your weight"))
weight_kg=weight*0.45
print(weight_kg)

# STRING

lang="python"
print(lang[:])
another=lang[:]
print(another)

name='maria'
print(name[-1])
print(name[1:-1])
first_name="Mark"
last_name="Warner"
msg=first_name + " " + last_name
print(msg)
print(f"{first_name} {last_name}")
name="M Mohaiminul Islam"
print(len(name))
print(name.find("m"))
print("Mohaiminul" in name)

# #MATH FUNCTION
a=4.6
print(round(a))
print(abs(a))
import math
a=4.6
print(math.floor(a))
print(math.floor(a))
print(math.sqrt(a))
print(math.pow(a,2))

#IF CONDITION
Car= 30000
has_good_credit=False 
down_payment=0
if has_good_credit:
    down_payment=Car*.2 
else:
    down_payment=Car*.5
print(f"Down Payment is {down_payment}")        

has_sufficient_funds=True
has_good_results=False
if has_sufficient_funds and has_good_results:
    print("Eligible")
else:
    print("Not Eligible")    


has_sufficient_funds=True
has_good_results=False
if has_sufficient_funds and not has_good_results:
    print("Eligible")
else:
    print("Not Eligible")    

#2D LIST
matrix=[
    [1,2,3],
    [4,5,6],
    [7,8,9]
]
# print(matrix[1][0])
for rows in matrix:
    for columns in rows:
        print(columns[0][0])

numbers=[11,11,22,6,9,10]
numbers.append(29)
numbers.append(50)
numbers.insert(1,44)
print(numbers.count(11))
numbers.sort()
numbers.reverse()
numbers2=numbers.copy()
print(numbers2)
print(numbers)
numbers=[1,1,3,3,7,6,3,4,6,7]
sole_numbers=[]
for number in numbers:
    if number not in sole_numbers:
        sole_numbers.append(number)
print(sole_numbers)         

#TUPLE (Cant add and remove value,its immutable)
numbers=(1,2,3,4,5,)
print(numbers)

#SET(Cant store duplicate value)
numbers={1,2,3,4,5,4}
print(numbers)

#Unpacking
coordinates=(1,2,3)
x=coordinates[0] 
y= coordinates[1]
z=coordinates[2]
print(x*y*z)
#we can solve this through unpacking
coordinate=(1,2,3)
x,y,z=coordinate 
print(x)
print(y)   
print(z)
print(x*y*z)   

#DICTIONARY
numbers=input("Type a Number to Convert: ")
digit_to_words={
    "0":"zero",
    "1":"One",
    "2":"Two",
    "3":"Three",
    "4":"Five"
}
output=""
for number in numbers:
    output+=digit_to_words.get(number, "...")+ " "
print(output)

# # # # # # #WHILE CONDITION
i=1
while i<10:

    print("*" * i)
    i+=1
# NESTED WHILE LOOP
i=1
while i<5:
    j=1
    while j<5:
        print(j)
        j=j+1
    i=i+1    

#FOR LOOP
language="Python"
for item in language:
    print(item, end='')

names=["niloy","robi",'nayan','mehedi','piyas']
for item in range(names):
    print(item)
price=[40,50,75,100]
total=0
for i in price:
    total=total+i
print("total is:", total)

#NESTED LOOP
for x in range(4):
    for y in range(3):
        print(f"{x},{y}")

numbers=[6,2,6,2,6]
for x in numbers:
    print("x"*x)
numbers=[6,2,6,2,6]
for x in numbers:
    output= ""
    for y in range(x):
        output+="X"
    print(output)    

# FUNCTION

def calc_age():
    birth_year=1994
    age=2023-birth_year
    print(f"Age is: {age}")

print("Name: M Mohaiminul Islam")
calc_age()
def calc_age(birth_year,current_year):
    
    age=current_year-birth_year
    print(f"Age is: {age}")

print("Name: M Mohaiminul Islam")
calc_age(current_year=1994,birth_year=2020)
def calc_age():
    birth_year=1994
    age=2023-birth_year
    return age

print("Name: M Mohaiminul Islam")
z=calc_age()
print(z)

#ERROR HANDLING
try:
  age1=int(input("what is your age: "))
  age2=int(input("what is your age: "))
  avg=age1/age2
  print(avg)
except:
  print("not valid")

# OOP
  
class Car:
    def brand(self):
        print("Audi")
    def color(self):
        print("Blue")   
    def move(self):
        print("Move")
car=Car()
car.brand()
car.color()   

#Constructor
class Car:
    def __init__(self,start,stop):
        self.start=start
        self.stop=stop
    def brand(self):
        print("Audi")
    def color(self):
        print("Blue")   
    def move(self):
        print("Move")
car=Car("Start","stop")
print(car.start)
car.brand()
# car.color() 
class Dog:
    def __init__(self,breed,bark):
        self.breed=breed
        self.brak=bark
    def introduction(self):
        print(f"breed is:{self.breed}" )
        print(f"breed is:{self.brak}" )

dog=Dog("German Shepard","Black")
dog.introduction()
#ENCAPSULATION
class Car:
    def __init__(self, make, model, year):
        self.__make = make    # private attribute
        self._model = model   # protected attribute
        self.year = year      # public attribute

    def drive(self):
        print("The car is driving.")

    def stop(self):
        print("The car is stopping.")

    def turn(self, direction):
        print(f"The car is turning {direction}.")

    def get_make(self):
        return self.__make   # private method

# Create an object of the Car class
my_car = Car("Toyota", "Camry", 2020)

# Access public attribute
print(my_car.year)  # Output: 2020

# Access protected attribute
print(my_car._model)  # Output: "Camry"

# Access private attribute
# This will raise an AttributeError as it's not directly accessible
print(my_car.__make)  # Raises AttributeError

# However, we can still access it using the name mangling syntax
print(my_car._Car__make)  # Output: "Toyota"

# Access private method
print(my_car.get_make())  # Output: "Toyota"

#INHERITANCE
#Single inheritance
class Animal:
    def speak(self):
        print("Animal speaks")

class Dog(Animal):
    pass

dog = Dog()
dog.speak()  # Output: "Animal speaks"
#Multiple inheritance
class Animal:
    def speak(self):
        print("Animal speaks")

class Mammal:
    def walk(self):
        print("Mammal walks")

class Dog(Animal, Mammal):
    pass

dog = Dog()
dog.speak()  # Output: "Animal speaks"
dog.walk()  # Output: "Mammal walks"

#Multi-Level Inheritance
class Animal:
    def speak(self):
        print("Animal speaks")

class Dog(Animal):
    pass

class Bulldog(Dog):
    pass

bulldog = Bulldog()
bulldog.speak()  # Output: "Animal speaks"

#Hierarchical inheritance
class Animal:
    def speak(self):
        print("Animal speaks")

class Dog(Animal):
    pass

class Cat(Animal):
    pass

dog = Dog()
cat = Cat()
dog.speak()  # Output: "Animal speaks"
cat.speak()  # Output: "Animal speaks"

#Hybrid inheritance: 
class Animal:
    def speak(self):
        print("Animal speaks")

class Mammal:
    def walk(self):
        print("Mammal walks")

class Dolphin(Animal, Mammal):
    pass

dolphin = Dolphin()
dolphin.speak()  # Output: "Animal speaks"
dolphin.walk()  # Output: "Mammal walks"

#POLYMORPHISM 
#METHOD OVERRIDING
class Animal:
    def speak(self):
        print("Animal speaks")

class Dog(Animal):
    def speak(self):
        print("Dog barks")

class Cat(Animal):
    def speak(self):
        print("Cat meows")

# Polymorphic usage of methods
animal = Animal()
dog = Dog()
cat = Cat()

animal.speak()  # Output: "Animal speaks"
dog.speak()     # Output: "Dog barks"
cat.speak()     # Output: "Cat meows"
                       