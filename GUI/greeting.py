from tkinter import *

class Application(Frame):
    def __init__(self, master):
        super(Application, self).__init__(master)
        self.create_widgets()
        self.grid()

    def create_widgets(self):
        self.welcome = Label(text="Selamat Pagi", font=("Arial", 20))
        self.welcome.grid(row=1, column=1)

        self.label_input = Label(text="masukan nama", font=("Arial", 10))
        self.label_input.grid(row=2, column=1)

        self.input = Entry()
        self.input.grid(row=3, column=1)
        
        self.button = Button(text="Submit", command=self.show_output)
        self.button.grid(row=5, column=1)

        self.output = Label()
        self.output.grid(row=6, column=1)

    def show_output(self):
        text = self.input.get()
        self.output['text'] = text

window = Tk()
window.title("Greeting to person :)")
window.geometry("300x150")

app = Application(window)
app.mainloop()