# This code is used to create Self Signed Certificate
from OpenSSL import crypto
from cryptography.fernet import Fernet
import hashlib
import random
import string
from threading import Lock


class SingletonMeta(type):
    """
    This is a thread-safe implementation of Singleton.
    """

    _instances = {}

    _lock: Lock = Lock()
    """
    We now have a lock object that will be used to synchronize threads during
    first access to the Singleton.
    """

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        # Now, imagine that the program has just been launched. Since there's no
        # Singleton instance yet, multiple threads can simultaneously pass the
        # previous conditional and reach this point almost at the same time. The
        # first of them will acquire lock and will proceed further, while the
        # rest will wait here.
        with cls._lock:
            # The first thread to acquire the lock, reaches this conditional,
            # goes inside and creates the Singleton instance. Once it leaves the
            # lock block, a thread that might have been waiting for the lock
            # release may then enter this section. But since the Singleton field
            # is already initialized, the thread won't create a new object.
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance

        return cls._instances[cls]


class Certificate(metaclass=SingletonMeta):

    fernet: Fernet = None

    def __init__(self, keys: dict) -> None:
        self.fernet = Fernet(key=Fernet.generate_key())
        self.fernet._signing_key = keys["_signing_key"]
        self.fernet._encryption_key = keys["_encryption_key"]

    def cert_gen(
        self,
        emailAddress="emailAddress",
        commonName="commonName",
        countryName="NT",
        localityName="localityName",
        stateOrProvinceName="stateOrProvinceName",
        organizationName="organizationName",
        organizationUnitName="organizationUnitName",
        serialNumber=0,
        validityStartInSeconds=0,
        validityEndInSeconds=10 * 365 * 24 * 60 * 60,
        KEY_FILE="private.key",
        CERT_FILE="selfsigned.crt",
    ):

        # Can look at generated file using OpenSSL
        # Openssl x509 -inform pem -in selfsigned.crt -noout -text
        # Create a key pair
        k = crypto.PKey()
        k.generate_key(crypto.TYPE_RSA, 4096)

        # Create a self-signed cert
        cert = crypto.X509()
        cert.get_subject().C = countryName
        cert.get_subject().ST = stateOrProvinceName
        cert.get_subject().L = localityName
        cert.get_subject().O = organizationName  # noqa: E741
        cert.get_subject().OU = organizationUnitName
        cert.get_subject().CN = commonName
        cert.get_subject().emailAddress = emailAddress
        cert.set_serial_number(
            self.serial_id_generator() if serialNumber == 0 else serialNumber
        )
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(validityEndInSeconds)
        cert.set_issuer(cert.get_subject())
        cert.set_pubkey(k)
        cert.sign(k, "sha512")

        with open(CERT_FILE, "wt") as f:
            f.write(crypto.dump_certificate(
                crypto.FILETYPE_PEM, cert).decode("utf-8"))

        with open(KEY_FILE, "wt") as f:
            f.write(crypto.dump_privatekey(
                crypto.FILETYPE_PEM, k).decode("utf-8"))

    def hash_file(self, filename):
        """
        This function is for the hashing concept.

        Actions:
            - Takes the hash of Existing Self-Signed Certificate to create session Key

        Returns::
            - Returns the SHA-1 hash of the file passed into it
        """

        # Make a hash object
        h = hashlib.sha1()

        # Open file for reading in binary mode
        with open(filename, "rb") as file:

            # Loop till the end of the file
            chunk = 0

            while chunk != b"":

                # Read only 1024 bytes at a time
                chunk = file.read(1024)
                h.update(chunk)

        # Return the hex representation of digest
        return h.hexdigest()

    def encrypt_with_fernet(self, message: bytes) -> bytes:
        """
        Use to encrypt session key
        """
        return self.fernet.encrypt(message)

    def decrypt_with_fernet(self, enc_message: bytes) -> bytes:
        """
        Used to decrypt session key
        """
        return self.fernet.decrypt(enc_message)

    def serial_id_generator(self, size: int = 10, chars: str = string.digits) -> str:
        """
        Generate a random string
        """
        return int("".join(random.choice(chars) for _ in range(size)))
