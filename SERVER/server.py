import socket
from sqlalchemy import text,create_engine

server=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

host,port='',8000

server.bind((host,port))

eng_postgre = create_engine('postgresql+psycopg2://postgres:fyVA3BcTOFENZydZ@localhost:5432/Latihan')

server.listen(5)
while True:
    print("Waitting for client to connect")
    client,addr = server.accept()
    print(client,addr)

    sql = client.recv(1024).decode()
    print(sql)

    client.send(str.encode('Connection is running'))
    with eng_postgre.connect() as con:
        messenger = con.execute(text(sql))
        data = messenger.fetchall()

        client.send(str.encode(str(data)))