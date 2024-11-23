class Car:
    def __init__(self,pintu,bangku):
        self.pintu=pintu
        self.roda=4
        self.bangku=bangku

    def action(self,name,speed):
        print(f"{name} sedang berjalan dengan kecepatan {speed} km/jam")

    def bensin(self):
        isi = input("Jenis Bensin yang digunakan : ")
        print(f"bensin yang diisi {isi} liter")

#inheritance
class Truck(Car):
    def __init__(self,pintu,roda,bangku):
        super(Truck,self).__init__(pintu,roda,bangku)
        self.roda=6

    def action(self):
        print("angkut beton")