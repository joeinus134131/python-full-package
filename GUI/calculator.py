from tkinter import *
from sqlalchemy import text
from sqlalchemy.engine import create_engine
import pandas as pd
import os

db_port=5432
# db_password=
eng_postgre = create_engine('postgresql+psycopg2://postgres:fyVA3BcTOFENZydZ@localhost:5432/Latihan')

class Application(Frame):
    def __init__(self, master):
        super(Application, self).__init__(master)
        self.create_widgets()
        self.grid()
        self.history_value = ''

    def create_widgets(self):
        self.welcome = Label(text="Selamat Pagi", font=("Arial", 20))
        self.welcome.grid(row=1, column=1)

        self.penjumlahan()
        self.perkalian()

        # history logs
        self.history_label = Label(text="History calculate")
        self.history_label.grid(row=11,columnspan=3)

        # button download
        self.btn_download = Button(text="Download data", command=self.download)
        self.btn_download.grid(row=13,column=3)

        # button show table
        self.btn_show = Button(text='Show Data from Database',command=self.set_table)
        self.btn_show.grid(row=15,column=3)

    
    def penjumlahan(self):
        self.input1 = Entry()
        self.input1.grid(row=3,column=1)

        self.tambah = Label(text="+")
        self.tambah.grid(row=3,column=2)

        self.input2 = Entry()
        self.input2.grid(row=3,column=3)

        self.button1  = Button(text="Hitung", command=self.cal_jumlah)
        self.button1.grid(row=4,column=1, columnspan=3)

        self.output1 = Label()
        self.output1.grid(row=5,column=2)

    def perkalian(self):
        self.input3 = Entry()
        self.input3.grid(row=6,column=1)

        self.kali = Label(text="x")
        self.kali.grid(row=6,column=2)

        self.input4 = Entry()
        self.input4.grid(row=6,column=3)

        self.button2  = Button(text="Hitung",command=self.cal_kali)
        self.button2.grid(row=7,column=1, columnspan=3)

        self.output2 = Label()
        self.output2.grid(row=8,column=2)

    def cal_jumlah(self):
        angka1 = int(self.input1.get())
        angka2 = int(self.input2.get())

        hasil = angka1+angka2
        self.output1['text']=hasil
        self.show_history(angka1,'+',angka2,hasil)

    def cal_kali(self):
        angka3 = int(self.input3.get())
        angka4 = int(self.input4.get())

        hasil1 = angka3*angka4
        self.output2['text']=hasil1
        self.show_history(angka3,'x',angka4,hasil1)

    def show_history(self,n1,operator,n2,hasil):
        self.history_value = self.history_value+f'{n1}{operator}{n2}={hasil}\n'
        self.history_label['text']=str(self.history_value)
        self.send_to_db(n1,operator,n2,hasil)

    def send_to_db(self,n1,operator,n2,hasil):
        with eng_postgre.connect() as con:
            con.execute(text(f'INSERT INTO calculate_logs VALUES {tuple([n1,operator,n2,hasil])}'))
            con.commit()

    def set_table(self):
        with eng_postgre.connect() as con:
            table = con.execute(text(f'SELECT * FROM calculate_logs')).fetchall()
        
        table.insert(0,['angka1','angka2','operator', 'hasil'])
        table_rows = len(table)
        table_column = len(table[0])

        for i in range(table_rows):
            for j in range(table_column):
                self.e = Label()
                self.e.grid(row=i+1,column=j+5)
                self.e['text'] = table[i][j]
    
    def download(self):
        df=pd.read_sql_query('SELECT * FROM calculate_logs',con=eng_postgre)
        path=os.getcwd()
        path=path.replace('\\','//')
        df.to_excel(f'C://Users/hp/Downloads/data_1.xlsx', index=False)
        self.history_label['text'] = 'File tersimpan sebagai data.xlsx'

window = Tk()
window.title("Simple Calculator")
window.geometry("300x150")

app = Application(window)
app.mainloop()