class Animal:
    def __init__(self, name, legs, color):
        self.name = name
        self.legs = legs
        self.color = color
Animal1 = Animal('Dog', 4, 'Brown')
Animal2 = Animal('Cat', 4, 'White')
print(Animal1.name)