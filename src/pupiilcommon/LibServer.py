#!/usr/bin/env python3

import face_recognition
import selectors
import secrets
import pathlib
import pickle
import socket
import struct
import json
import sys
import csv
import cv2
import io
import os

from . import Certificate as cert

request_search = {
    "Example 1": "Follow the white rabbit. \U0001f430",
    "Example 2": "In the caves beneath the Misty Mountains. \U0001f48d",
    "\U0001f436": "\U0001f43e Playing ball! \U0001f3d0",
}


class Message:
    def __init__(self, selector, sock, addr):
        self.selector = selector
        self.sock = sock
        self.addr = addr
        self._recv_buffer = b""
        self._send_buffer = b""
        self._jsonheader_len = None
        self.jsonheader = None
        self.request = None
        self.response_created = False
        self._registered_addresses = []
        self._cert_gen_random_data = self._get_cert_gen_csv_data()
        self._certificate_aux = cert.Certificate(self._get_key_response_from_ckms())
        self.detector = cv2.CascadeClassifier(f"{pathlib.Path(__file__).parent.absolute()}/data/HaarCascade.xml")
        self.encodings = pickle.loads(open(f"{pathlib.Path(__file__).parent.absolute()}/data/PREncodings.pkl", "rb").read())

    def _get_key_response_from_ckms(self) -> dict:

        HOST = "127.52.0.2"  # The server's hostname or IP address
        PORT = 6000  # The port used by the CKMS

        keys = {}

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ckms_socket:
            ckms_socket.connect((HOST, PORT))
            ckms_socket.sendall(
                f"Connection from {self.addr}:{self.sock}".encode("utf-8")
            )
            keys = ckms_socket.recv(1024)

        return dict(_signing_key=keys[:16], _encryption_key=keys[17:])

    def _detect_faces(self, frame):
        # convert the input frame from (1) BGR to grayscale (for face
        # detection) and (2) from BGR to RGB (for face recognition)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # detect faces in the grayscale frame
        rects = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # OpenCV returns bounding box coordinates in (x, y, w, h) order
        # but we need them in (top, right, bottom, left) order, so we
        # need to do a bit of reordering
        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

        # compute the facial embeddings for each face bounding box
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(
                self.encodings["encodings"], encoding, tolerance=0.46
            )
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = self.encodings["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)

        return boxes, names

    def _get_cert_gen_csv_data(self):

        csv_data = dict(email_address=[], common_name=[], country_name=[])

        with open(f"{pathlib.Path(__file__).parent.absolute()}/data/CertGenData.csv") as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=",")
            line_count = 0

            for row in csv_reader:

                if line_count == 0:
                    print(f'[LIBSERVER::MESSAGE::_GET_CERT_GEN_CSV_DATA] Column names are {", ".join(row)}')
                else:
                    csv_data["email_address"].append(row[0])
                    csv_data["common_name"].append(row[1])
                    csv_data["country_name"].append(row[2])

                line_count += 1

            print(f"[LIBSERVER::MESSAGE::_GET_CERT_GEN_CSV_DATA] Processed {line_count} lines.")

        return csv_data

    def _set_selector_events_mask(self, mode):
        """Set selector to listen for events: mode is 'r', 'w', or 'rw'."""
        if mode == "r":
            events = selectors.EVENT_READ
        elif mode == "w":
            events = selectors.EVENT_WRITE
        elif mode == "rw":
            events = selectors.EVENT_READ | selectors.EVENT_WRITE
        else:
            raise ValueError(f"Invalid events mask mode {repr(mode)}.")
        self.selector.modify(self.sock, events, data=self)

    def _read(self):
        try:
            # Should be ready to read
            data = self.sock.recv(4096)
        except BlockingIOError:
            # Resource temporarily unavailable (errno EWOULDBLOCK)
            pass
        else:
            if data:
                self._recv_buffer += data
            else:
                raise RuntimeError("Peer closed.")

    def _write(self):
        if self._send_buffer:
            print("[LIBSERVER::MESSAGE::_WRITE] Sending", repr(self._send_buffer), "to", self.addr)
            try:
                # Should be ready to write
                sent = self.sock.send(self._send_buffer)
            except BlockingIOError:
                # Resource temporarily unavailable (errno EWOULDBLOCK)
                pass
            else:
                self._send_buffer = self._send_buffer[sent:]
                # # Close when the buffer is drained. The response has been sent.
                # if sent and not self._send_buffer:
                #     self.close()

    def _json_encode(self, obj, encoding):
        return json.dumps(obj, ensure_ascii=False).encode(encoding)

    def _json_decode(self, json_bytes, encoding):
        tiow = io.TextIOWrapper(io.BytesIO(json_bytes), encoding=encoding, newline="")
        obj = json.load(tiow)
        tiow.close()
        return obj

    def _create_message(self, *, content_bytes, content_type, content_encoding):
        jsonheader = {
            "byteorder": sys.byteorder,
            "content-type": content_type,
            "content-encoding": content_encoding,
            "content-length": len(content_bytes),
        }
        jsonheader_bytes = self._json_encode(jsonheader, "utf-8")
        message_hdr = struct.pack(">H", len(jsonheader_bytes))
        message = message_hdr + jsonheader_bytes + content_bytes
        return message

    def _create_response_json_content(self):
        action = self.request.get("action")
        content_encoding = "utf-8"
        if action == "search":
            query = self.request.get("value")
            answer = request_search.get(query) or f'No match for "{query}".'
            content = {"result": answer}
        elif action == "add":

            if not os.path.exists("certs"):
                os.makedirs("certs")

            hostname = self.request.get("value")["hostname"]
            addr = self.request.get("value")["addr"]
            port = self.addr[1]
            addr_mac = self.request.get("value")["addr_mac"]
            peer_mac = self.request.get("value")["peer_mac"]

            content = {}

            if addr != "127.0.0.1":
                content = {"result": "invalid IP"}
            else:
                PATH = f"certs/{addr_mac}-{self.addr[0]}-{port}"

                os.makedirs(PATH)

                self._certificate_aux.cert_gen(
                    emailAddress=secrets.choice(
                        self._cert_gen_random_data["email_address"]
                    ),
                    commonName=secrets.choice(
                        self._cert_gen_random_data["common_name"]
                    ),
                    countryName=secrets.choice(
                        self._cert_gen_random_data["country_name"]
                    ),
                    localityName=secrets.token_urlsafe(nbytes=10),
                    stateOrProvinceName=secrets.token_urlsafe(nbytes=10),
                    organizationName=secrets.token_urlsafe(nbytes=10),
                    organizationUnitName=secrets.token_urlsafe(nbytes=10),
                    KEY_FILE=f"{PATH}/private.key",
                    CERT_FILE=f"{PATH}/selfsigned.crt",
                )

                hash_file_data = self._certificate_aux.hash_file(
                    f"{PATH}/selfsigned.crt"
                )

                result_data_to_store = dict(
                    hostname=hostname,
                    addr_ip=addr,
                    port=port,
                    addr_mac=addr_mac,
                    peer_mac=peer_mac,
                )

                self._registered_addresses.append(
                    self._certificate_aux.encrypt_with_fernet(
                        self._json_encode(result_data_to_store, content_encoding)
                    )
                )

                content = {
                    "result": f"successfully added client {self.addr, port}",
                    "hashed_cert": hash_file_data,
                }
        else:
            content = {"result": f'Error: invalid action "{action}".'}

        return {
            "content_bytes": self._certificate_aux.encrypt_with_fernet(
                self._json_encode(content, content_encoding)
            ),
            "content_type": "text/json",
            "content_encoding": content_encoding,
        }

    def _create_response_binary_content(self):
        return {
            "content_bytes": b"First 10 bytes of request: " + self.request[:10],
            "content_type": "binary/custom-server-binary-type",
            "content_encoding": "binary",
        }

    def process_events(self, mask):
        if mask & selectors.EVENT_READ:
            self.read()
        if mask & selectors.EVENT_WRITE:
            self.write()

    def read(self):
        self._read()

        if self._jsonheader_len is None:
            self.process_protoheader()

        if self._jsonheader_len is not None:
            if self.jsonheader is None:
                self.process_jsonheader()

        if self.jsonheader:
            if self.request is None:
                self.process_request()

    def write(self):
        if self.request:
            if not self.response_created:
                self.create_response()

        self._write()

    def close(self):
        print("[LIB_SERVER] Closing connection to", self.addr)
        try:
            self.selector.unregister(self.sock)
        except Exception as e:
            print(
                "[LIB_SERVER] Error: selector.unregister() exception for",
                f"{self.addr}: {repr(e)}",
            )

        try:
            self.sock.close()
        except OSError as e:
            print(
                "[LIB_SERVER] Error: socket.close() exception for",
                f"{self.addr}: {repr(e)}",
            )
        finally:
            # Delete reference to socket object for garbage collection
            self.sock = None

    def process_protoheader(self):
        hdrlen = 2
        if len(self._recv_buffer) >= hdrlen:
            self._jsonheader_len = struct.unpack(">H", self._recv_buffer[:hdrlen])[0]
            self._recv_buffer = self._recv_buffer[hdrlen:]

    def process_jsonheader(self):
        hdrlen = self._jsonheader_len
        if len(self._recv_buffer) >= hdrlen:
            self.jsonheader = self._json_decode(self._recv_buffer[:hdrlen], "utf-8")
            self._recv_buffer = self._recv_buffer[hdrlen:]
            for reqhdr in (
                "byteorder",
                "content-length",
                "content-type",
                "content-encoding",
            ):
                if reqhdr not in self.jsonheader:
                    raise ValueError(f'Missing required header "{reqhdr}".')

    def process_request(self):
        content_len = self.jsonheader["content-length"]
        if not len(self._recv_buffer) >= content_len:
            return
        data = self._recv_buffer[:content_len]
        self._recv_buffer = self._recv_buffer[content_len:]
        if self.jsonheader["content-type"] == "text/json":
            encoding = self.jsonheader["content-encoding"]
            self.request = self._json_decode(data, encoding)
            print("[LIB_SERVER] Received request", repr(self.request), "from", self.addr)
        else:
            # Binary or unknown content-type
            self.request = data
            print(
                f'[LIB_SERVER] Received {self.jsonheader["content-type"]} request from',
                self.addr,
            )
        # Set selector to listen for write events, we're done reading.
        self._set_selector_events_mask("w")

    def create_response(self):
        if self.jsonheader["content-type"] == "text/json":
            response = self._create_response_json_content()
        else:
            # Binary or unknown content-type
            response = self._create_response_binary_content()
        message = self._create_message(**response)
        self.response_created = True
        self._send_buffer += message
