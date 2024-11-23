import socket

client=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

host,port=socket.gethostbyname('localhost'),8000

client.connect((host,port))
client.send(str.encode('SELECT * FROM calculate_logs'))
while True:
    print(client.recv(1024).decode())