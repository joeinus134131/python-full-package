import tkinter as tk
from tkinter import messagebox

#fungsi untuk menghitung usia
def hitung_usia():
    try:
        tahun_lahir = int(entry_tahun_lahir.get())
        tahun_sekarang = 2024
        usia = tahun_sekarang - tahun_lahir
        label_usia.config(text="Usia Anda: " + str(usia))
    except ValueError:
        messagebox.showerror(title="Error",message="Masukan tahun lahir yang valid!")

age = tk.Tk()
age.title("Hitung Usia")
age.geometry("300x150")

#label untuk judul
judul = tk.Label(age, text="Aplikasi Hitung Usia", font=("Arial", 20))
judul.pack(pady=10)

#input tahun lahir
label_tahun_lahir = tk.Label(age,text="Masukan tahun lahir anda : ")
label_tahun_lahir.pack()

entry_tahun_lahir = tk.Entry(age, width=20)
entry_tahun_lahir.pack()

#add button submit
button_submit = tk.Button(age, text="Hitung Usia", command=hitung_usia)
button_submit.pack()

#label hasil perhitungan
label_usia = tk.Label(age,text="",font=("Arial", 12))
label_usia.pack()

age.mainloop()