# Classes -- Real world entities or objects
# Class -- Blueprint of an object
class Car:
    pass

car1 = Car()
car2 = Car()

print(car1)

car1.windows = 5
car1.tyres = 4
car1.engine = 'Diesel'
car2.windows = 3
car2.tyres = 4
car2.engine = 'Petrol'
car2.color = 'Red'

print(car1.windows)
print(car2.engine)
print(car2.color)