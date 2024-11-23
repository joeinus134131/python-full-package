import tkinter as tk

app = tk.Tk()
app.title("Absensi")
app.geometry("300x150")

judul = tk.Label(app, text="Selamat Pagi", font=("Arial", 20))
judul.pack(pady=10)

label_inputan = tk.Label(app, text="masukan nama", font=("Arial", 10))
label_inputan.pack()

inputan_nama = tk.Entry(app,width=20)
inputan_nama.pack()

button_submit = tk.Button(app, text="Submit", command=lambda: print("nama kamu adalah", inputan_nama.get()))
button_submit.pack()

app.mainloop()