class Orang:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def category(self):
        if 0<self.age<10:
            return "Kids"
        elif 10<=self.age<18:
            return "Teenagers"
        elif 18<=self.age<65:
            return "Adults"
        else:
            return "Senior"