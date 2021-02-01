import serial


port = 'com3'
baudrate = 115200

ser = serial.Serial(port,baudrate, timeout=0.5,bytesize=8,parity='N',stopbits=1,
                    xonxoff=False,rtscts=False,dsrdtr=False,interCharTimeout=None)

with open('1.txt', 'w') as f:
    sum_str = ''
    while True:
        str = ser.read(2000)
        a = str.decode("ascii")
        a = a.replace('\r\n', '')
        sum_str = sum_str + a
        f.write(sum_str)
        print(sum_str)



