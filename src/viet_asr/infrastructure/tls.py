"""HTTPS için kendinden imzalı sertifika (kapalı ağ: dışarıdan sertifika alınamaz).

Tarayıcılar mikrofonu yalnızca güvenli sayfada açar: http://localhost olur, ama başka cihazdan
http://192.168.1.20:8001 olmaz. Bu yüzden başka cihazdan canlı altyazı için https gerekir.
Sertifika bu bilgisayarın adlarını ve ağ adreslerini içerir; adres değişirse yeniden üretilir.
İlk açılışta tarayıcı "güvenli değil" uyarısı verir: "Gelişmiş > yine de devam et" bir kez yeterli.
"""
from __future__ import annotations

import datetime as dt
import ipaddress
import socket
from pathlib import Path


def local_addresses() -> list[str]:
    """Bu bilgisayarın IPv4 adresleri (127.0.0.1 dahil). İnternet gerekmez: UDP 'bağlanma'
    paket göndermez, yalnızca hangi arayüzün kullanılacağını sorar."""
    found = {"127.0.0.1"}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(info[4][0])
    except OSError:
        pass
    for probe in ("10.255.255.255", "192.168.255.255", "172.31.255.255"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((probe, 1))
                found.add(s.getsockname()[0])
        except OSError:
            pass
    return sorted(a for a in found if not a.startswith("0."))


def _names() -> tuple[list[str], list[str]]:
    dns = sorted({"localhost", socket.gethostname()} - {""})
    return dns, local_addresses()


def _covers(cert_path: Path, dns: list[str], ips: list[str]) -> bool:
    from cryptography import x509

    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except Exception:
        return False
    have_ip = {str(a) for a in san.get_values_for_type(x509.IPAddress)}
    have_dns = set(san.get_values_for_type(x509.DNSName))
    fresh = cert.not_valid_after_utc > dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=7)
    return fresh and set(ips) <= have_ip and set(dns) <= have_dns


def ensure_self_signed(directory: str | Path) -> tuple[str, str]:
    """directory/cert.pem ve key.pem: yoksa, süresi bitiyorsa ya da ağ adresi değiştiyse üretir.
    -> (sertifika yolu, anahtar yolu)"""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    d = Path(directory)
    cert_path, key_path = d / "cert.pem", d / "key.pem"
    dns, ips = _names()
    if cert_path.exists() and key_path.exists() and _covers(cert_path, dns, ips):
        return str(cert_path), str(key_path)

    d.mkdir(parents=True, exist_ok=True)
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Ses Analizi (yerel)")])
    now = dt.datetime.now(dt.timezone.utc)
    san = [x509.DNSName(n) for n in dns] + [x509.IPAddress(ipaddress.ip_address(a)) for a in ips]
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=825))
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .sign(key, hashes.SHA256()))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                           serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return str(cert_path), str(key_path)
