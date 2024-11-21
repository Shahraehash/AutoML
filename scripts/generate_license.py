import datetime
from getpass import getpass
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
import uuid

from jose import jwt

private_key_password = getpass("Enter private key password:")
days_valid = input("Enter number of days license will be valid (starting from today):")

if not days_valid:
    days_valid = 365
else:
    days_valid = int(days_valid)

expiration_date = datetime.datetime.now() + datetime.timedelta(days=days_valid)
payload = {"expiration_date": expiration_date.isoformat()}
with open("private_key.pem", "rb") as key_file:
    private_key = serialization.load_pem_private_key(
        key_file.read(),
        password=private_key_password.encode(),
        backend=default_backend(),
    )

private_key_pem = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
)
license_key = jwt.encode(payload, private_key_pem, algorithm="RS256")
with open(f"{uuid.uuid4()}.key", "w") as license_file:
    license_file.write(license_key)
