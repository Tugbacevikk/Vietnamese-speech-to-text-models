"""SDP (RFC 4566 / RFC 3551): RTP akışının nerede ve hangi biçimde geleceğini tarif eden metin.

SIP ile kurulan her görüşmede santral bu metni verir:

    v=0
    o=- 0 0 IN IP4 10.0.0.5
    s=cagri
    c=IN IP4 10.0.0.5
    t=0 0
    m=audio 5004 RTP/AVP 96 0
    a=rtpmap:96 L16/16000/1
    a=rtpmap:0 PCMU/8000
    a=ptime:20

"m=audio 5004 RTP/AVP 96 0": ses 5004 portuna, yük türü 96 ya da 0 olarak gelir.
"a=rtpmap:96 L16/16000/1": 96 = ham 16 bit PCM (WAV'ın başlıksız hali), 16 kHz, tek kanal.
Opus (OGG'nin içindeki ses) "a=rtpmap:111 opus/48000/2" diye yazılır.

Bu modül yalnızca çözümler; hangi alıcının kullanılacağına streams.py karar verir:
  PCMU, PCMA, L16  -> bizim RTP alıcımız (ek kütüphane yok)
  opus, G722, ...  -> ffmpeg (SDP dosyasıyla)
"""
from __future__ import annotations

from dataclasses import dataclass, field

# RFC 3551 sabit yük türleri (rtpmap satırı yazılmasa da geçerli)
STATIC = {0: ("PCMU", 8000, 1), 8: ("PCMA", 8000, 1), 9: ("G722", 8000, 1),
          10: ("L16", 44100, 2), 11: ("L16", 44100, 1)}
NATIVE = {"PCMU": "pcmu", "PCMA": "pcma", "L16": "l16"}


class SdpError(ValueError):
    pass


@dataclass(frozen=True)
class RtpFormat:
    pt: int
    encoding: str      # büyük harf: PCMU, PCMA, L16, OPUS, G722...
    clock_rate: int
    channels: int = 1

    @property
    def native(self) -> str | None:
        """Bizim alıcının çözebildiği adı (pcmu/pcma/l16) ya da None (ffmpeg gerekir)."""
        return NATIVE.get(self.encoding)


@dataclass(frozen=True)
class AudioMedia:
    port: int
    address: str | None            # c= satırındaki adres (yoksa None: her arayüz)
    formats: tuple[RtpFormat, ...]  # m= satırındaki sırayla (ilki tercih edilen)
    ptime: int | None = None
    direction: str = "sendrecv"
    attrs: dict[str, str] = field(default_factory=dict)

    def preferred(self) -> RtpFormat:
        """Önce bizim çözebildiğimiz ilk biçim, yoksa listedeki ilk biçim."""
        for f in self.formats:
            if f.native:
                return f
        return self.formats[0]


def parse(text: str) -> AudioMedia:
    """SDP metninden ilk ses (m=audio) bölümü. Ses yoksa SdpError."""
    session_addr: str | None = None
    media: list[dict] = []
    cur: dict | None = None
    for raw in text.replace("\r", "").split("\n"):
        line = raw.strip()
        if len(line) < 2 or line[1] != "=":
            continue
        k, v = line[0], line[2:].strip()
        if k == "m":
            parts = v.split()
            if len(parts) < 4:
                raise SdpError(f"bozuk m= satırı: {line}")
            cur = {"type": parts[0], "port": parts[1], "proto": parts[2], "pts": parts[3:],
                   "rtpmap": {}, "attrs": {}, "addr": None}
            media.append(cur)
        elif k == "c":
            addr = v.split()[-1].split("/")[0] if v.split() else None
            if cur is None:
                session_addr = addr
            else:
                cur["addr"] = addr
        elif k == "a" and cur is not None:
            name, _, val = v.partition(":")
            if name == "rtpmap":
                pt, _, desc = val.partition(" ")
                cur["rtpmap"][pt.strip()] = desc.strip()
            else:
                cur["attrs"][name] = val
    audio = next((m for m in media if m["type"] == "audio"), None)
    if audio is None:
        raise SdpError("SDP'de ses bölümü (m=audio) yok")
    if "RTP" not in audio["proto"].upper():
        raise SdpError(f"desteklenmeyen taşıma: {audio['proto']} (RTP/AVP beklenir)")
    try:
        port = int(audio["port"].split("/")[0])
    except ValueError as e:
        raise SdpError(f"geçersiz port: {audio['port']}") from e
    formats = []
    for pt_s in audio["pts"]:
        try:
            pt = int(pt_s)
        except ValueError:
            continue
        desc = audio["rtpmap"].get(pt_s)
        if desc:
            bits = desc.split("/")
            enc = bits[0].upper()
            try:
                rate = int(bits[1]) if len(bits) > 1 else 8000
                ch = int(bits[2]) if len(bits) > 2 and bits[2] else 1
            except ValueError as e:
                raise SdpError(f"bozuk rtpmap: {desc}") from e
            formats.append(RtpFormat(pt, enc, rate, ch))
        elif pt in STATIC:
            formats.append(RtpFormat(pt, *STATIC[pt]))
        # tanımsız dinamik yük türü (ör. telephone-event için rtpmap'siz 101): atlanır
    formats = [f for f in formats if f.encoding != "TELEPHONE-EVENT"] or formats
    if not formats:
        raise SdpError("ses biçimi tanımlı değil (rtpmap satırı yok)")
    direction = next((d for d in ("sendonly", "recvonly", "inactive", "sendrecv") if d in audio["attrs"]),
                     "sendrecv")
    ptime = audio["attrs"].get("ptime")
    return AudioMedia(port=port, address=audio["addr"] or session_addr, formats=tuple(formats),
                      ptime=int(ptime) if ptime and ptime.isdigit() else None, direction=direction,
                      attrs=dict(audio["attrs"]))


def write(port: int, fmt: RtpFormat, address: str = "127.0.0.1", ptime: int = 20) -> str:
    """Göndericinin (ör. scripts/rtp_gonder.py) dinleyiciye vereceği SDP."""
    ch = f"/{fmt.channels}" if fmt.channels != 1 else ""
    return (f"v=0\r\no=- 0 0 IN IP4 {address}\r\ns=viet-asr\r\nc=IN IP4 {address}\r\nt=0 0\r\n"
            f"m=audio {port} RTP/AVP {fmt.pt}\r\na=rtpmap:{fmt.pt} {fmt.encoding}/{fmt.clock_rate}{ch}\r\n"
            f"a=ptime:{ptime}\r\na=sendonly\r\n")
