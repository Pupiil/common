import socket
import netifaces


class MacAux:
    @staticmethod
    def get_machine_info():

        hostname = socket.gethostname()
        network_data = netifaces.ifaddresses("lo")[2][0]
        mac_data = netifaces.ifaddresses("lo")[17][0]

        return dict(
            hostname=hostname,
            host_ip=socket.gethostbyname(hostname),
            addr=network_data["addr"],
            netmask=network_data["netmask"],
            peer=network_data["peer"],
            addr_mac=mac_data["addr"],
            peer_mac=mac_data["peer"],
        )

    @staticmethod
    def get_remote_machine_info():
        remote_host = "www.python.org"
        try:
            print(
                "IP address of %s: %s"
                % (remote_host, socket.gethostbyname(remote_host))
            )
        except socket.error as err_msg:
            print("%s: %s" % (remote_host, err_msg))
