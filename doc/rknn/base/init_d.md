# base


## misc

### io

/usr/bin/io

``` bash
[root@M1808 smswitch]# io --help
Unknown option: ?
Raw memory i/o utility - $Revision: 1.5 $

io -v -1|2|4 -r|w [-l <len>] [-f <file>] <addr> [<value>]

    -v         Verbose, asks for confirmation
    -1|2|4     Sets memory access size in bytes (default byte)
    -l <len>   Length in bytes of area to access (defaults to
               one access, or whole file length)
    -r|w       Read from or Write to memory (default read)
    -f <file>  File to write on memory read, or
               to read on memory write
    <addr>     The memory address to access
    <val>      The value to write (implies -w)

Examples:
    io 0x1000                  Reads one byte from 0x1000
    io 0x1000 0x12             Writes 0x12 to location 0x1000
    io -2 -l 8 0x1000          Reads 8 words from 0x1000
    io -r -f dmp -l 100 200    Reads 100 bytes from addr 200 to file
    io -w -f img 0x10000       Writes the whole of file to memory

Note access size (-1|2|4) does not apply to file based accesses.
```

#### 使能网络通信
`io -4 -w 0xfe000900 0xffff050a`


## startup
### /etc/init.d/

/etc/init.d/ 是linux系统的启动项配置目录，含有所有的启动脚本。

```
[root@M1808 init.d]# ls /etc/init.d
rcK*  S01logging*  S10udev*     S21mountall.sh*  S30rpcbind*    S40network*   S41dhcpcd*  S49ntp*       S50sshd*       S60NPU_init*  S70vsftpd*   S80tftpd-hpa*    S99_auto_reboot*
rcS*  S10init*     S20urandom*  S30dbus*         S38_4g_reset*  S40rkisp_3A*  S41ipaddr*  S50launcher*  S50usbdevice*  S60openvpn*   S80dnsmasq*  S90Manufacture*  S99input-event-daemon*

```

### /etc/init.d/rcS
/etc/init.d/rcS


``` bash
#!/bin/sh


# Start all init scripts in /etc/init.d
# executing them in numerical order.
#
for i in /etc/init.d/S??* ;do

     # Ignore dangling symlinks (if any).
     [ ! -f "$i" ] && continue

     case "$i" in
        *.sh)
            # Source shell script for speed.
            (
                trap - INT QUIT TSTP
                set start
                . $i
            )
            ;;
        *)
            # No sh extension, so fork subprocess.
            $i start
            ;;
    esac
done
```

sh 是以配置脚本的形式启动，可以加快生成速度。
S50sshd 这种是以子进程的形式启动。

### /etc/init.d/S50sshd

``` bash
#!/bin/sh
#
# sshd        Starts sshd.
#

# Make sure the ssh-keygen progam exists
[ -f /usr/bin/ssh-keygen ] || exit 0

umask 077

start() {
        # Create any missing keys
        /usr/bin/ssh-keygen -A

        printf "Starting sshd: "
        /usr/sbin/sshd
        touch /var/lock/sshd
        echo "OK"
}
stop() {
        printf "Stopping sshd: "
        killall sshd
        rm -f /var/lock/sshd
        echo "OK"
}
restart() {
        stop
        start
}

case "$1" in
  start)
        start
        ;;
  stop)
        stop
        ;;
  restart|reload)
        restart
        ;;
  *)
        echo "Usage: $0 {start|stop|restart}"
        exit 1
esac

exit $?
```

### /etc/init.d/S40network
``` bash
#!/bin/sh
#
# Start the network....
#

# Debian ifupdown needs the /run/network lock directory
mkdir -p /run/network

case "$1" in
  start)
        printf "Starting network: "
        /sbin/ifup -a
        [ $? = 0 ] && echo "OK" || echo "FAIL"
        ;;
  stop)
        printf "Stopping network: "
        /sbin/ifdown -a
        [ $? = 0 ] && echo "OK" || echo "FAIL"
        ;;
  restart|reload)
        "$0" stop
        "$0" start
        ;;
  *)
        echo "Usage: $0 {start|stop|restart}"
        exit 1
esac

exit $?
```


### /etc/init.d/S999start_network

``` bash
vim /etc/init.d/S999start_network
chmod +x /etc/init.d/S999start_network

```

内容如下
``` bash
#!/bin/sh
#
# Start the io_network....
#

case "$1" in
  start)
        printf "Starting io network: \n"
        io -4 -w 0xfe000900 0xffff050a
	  # ping 192.168.1.101 
        # ifconfig eth0 192.168.1.136 netmask 255.255.255.0
        ;;
  stop)
        printf "Stopping io network: \n"
        ;;
  restart|reload)
        "$0" stop
        "$0" start
        ;;
  *)
        echo "Usage: $0 {start|stop|restart}"
        exit 1
esac

exit $?
```