class Animal:
    def __init__(self, name):
        self.name = name
        
    def speak(self):
        return f"{self.name} Rorrs"



# Creating a child class which means Animal class attributes will inherit into Dog class
# thats why even though we dont have name defined in dog class still its able to use it bcoz its defined in Animal class
# but both have same method `speak` then child method will execute in child object & overrides parent class method

class Dog(Animal):
    def speak1(self):
        return f"{self.name} barks"
        # return super().speak() + " and barks"1

# Creating an instance of Animal
lion = Animal("Lion")
print(lion.speak())

# Creating an instance of Dog
dog = Dog("Dog")
print(dog.speak())
