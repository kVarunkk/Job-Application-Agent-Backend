import os
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


def decrypt_aes_key(encrypted_aes_key_hex: str) -> bytes:
    """
    Decrypts AES key encrypted with KEK using AES-256-ECB.
    """
    kek = os.getenv("KEK_SECRET")
    if not kek:
        raise ValueError("KEK_SECRET not found in environment variables.")

    kek_bytes = bytes.fromhex(kek)
    encrypted_aes_key = bytes.fromhex(encrypted_aes_key_hex)

    cipher = Cipher(algorithms.AES(kek_bytes), modes.ECB(), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_key = decryptor.update(encrypted_aes_key) + decryptor.finalize()

    return decrypted_key  # Should be 32 bytes



def decrypt_password(encrypted_password_b64: str, aes_key: bytes) -> str:
    """
    Decrypts the Y Combinator password using the decrypted AES key.
    """
    encrypted_password = base64.b64decode(encrypted_password_b64)

    iv = encrypted_password[:16]
    ciphertext = encrypted_password[16:]

    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_password = decryptor.update(ciphertext) + decryptor.finalize()

    # Remove padding (PKCS7)
    pad_len = padded_password[-1]
    password = padded_password[:-pad_len]
    return password.decode("utf-8")
