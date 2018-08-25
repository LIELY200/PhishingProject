all: udp-server udp-client

udp-server: Udp-Server.c
	gcc -o Udp-Server Udp-Server.c

udp-client: Udp-Client.c
	gcc -o Udp-Client Udp-Client.c

clean:
	rm -f *.o Udp-Server Udp-Client

runs:
	./Udp-Server

runc:
	./Udp-Client

runs-strace:
	strace -f ./Udp-Server

runc-strace:
	strace -f ./Udp-Ccdlient