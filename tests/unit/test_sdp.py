"""SDP ile canlı akış: biçim/port SDP'den; L16 (WAV'ın ham hali) bizim alıcı, Opus (OGG) ffmpeg."""
import shutil
import socket
import struct
import subprocess
import threading
import time

import numpy as np
import pytest

from viet_asr.infrastructure import sdp
from viet_asr.infrastructure.ffmpeg_source import FfmpegSource
from viet_asr.infrastructure.rtp import RtpSource
from viet_asr.interface.streams import StreamError, StreamSpec, parse_source

SANTRAL = """v=0\r
o=asterisk 123 456 IN IP4 10.0.0.5\r
s=Asterisk\r
c=IN IP4 10.0.0.5\r
t=0 0\r
m=audio 18044 RTP/AVP 97 0 8 101\r
a=rtpmap:97 L16/16000/2\r
a=rtpmap:0 PCMU/8000\r
a=rtpmap:8 PCMA/8000\r
a=rtpmap:101 telephone-event/8000\r
a=fmtp:101 0-16\r
a=ptime:20\r
a=sendrecv\r
"""


def test_sdp_cozumleme():
    m = sdp.parse(SANTRAL)
    assert (m.port, m.address, m.ptime, m.direction) == (18044, "10.0.0.5", 20, "sendrecv")
    assert [(f.pt, f.encoding, f.clock_rate, f.channels) for f in m.formats] == [
        (97, "L16", 16000, 2), (0, "PCMU", 8000, 1), (8, "PCMA", 8000, 1)]   # tuş sesi ayıklandı
    assert m.preferred().native == "l16"
    # rtpmap yazılmamış sabit yük türü
    assert sdp.parse("v=0\nc=IN IP4 1.2.3.4\nm=audio 4000 RTP/AVP 0\n").formats[0].encoding == "PCMU"
    opus = sdp.parse("v=0\nm=audio 5020 RTP/AVP 111\na=rtpmap:111 opus/48000/2\n")
    assert opus.preferred().encoding == "OPUS" and opus.preferred().native is None
    for bad in ("v=0\nm=video 5000 RTP/AVP 96\n", "v=0\nm=audio x RTP/AVP 0\n", "v=0\n"):
        with pytest.raises(sdp.SdpError):
            sdp.parse(bad)
    w = sdp.write(5004, sdp.RtpFormat(96, "L16", 16000))
    assert sdp.parse(w).formats[0] == sdp.RtpFormat(96, "L16", 16000, 1)


def test_sdp_ile_kaynak_secimi(tmp_path):
    s = parse_source(StreamSpec(kaynak="rtp://127.0.0.1:0", sdp=SANTRAL), False)
    assert isinstance(s, RtpSource) and s.formats[97] == ("l16", 16000, 2) and s.port > 0
    s.sock.close()
    o = parse_source(StreamSpec(kaynak="", sdp="v=0\nm=audio 5020 RTP/AVP 111\na=rtpmap:111 opus/48000/2\n"),
                     False, tmp_path)
    assert isinstance(o, FfmpegSource) and o.port == 5020 and o.cleanup.endswith(".sdp")
    with pytest.raises(StreamError):
        parse_source(StreamSpec(kaynak="", sdp="v=0\n"), False)
    with pytest.raises(StreamError):  # ffmpeg biçiminde port SDP'den farklı olamaz
        parse_source(StreamSpec(kaynak="rtp://0.0.0.0:6000",
                                sdp="v=0\nm=audio 5020 RTP/AVP 111\na=rtpmap:111 opus/48000/2\n"), False)


def test_l16_stereo_sdp_ile_alinir_tus_sesi_atilir():
    src = parse_source(StreamSpec(kaynak="rtp://127.0.0.1:0", sdp=SANTRAL, idle_timeout=0.5), False)
    t = np.arange(16000) / 16000
    left, right = 0.4 * np.sin(2 * np.pi * 200 * t), 0.2 * np.sin(2 * np.pi * 200 * t)
    inter = np.stack([left, right], 1).reshape(-1)                      # LRLR...
    pcm = (inter * 32767).astype(">i2").tobytes()

    def send():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        step = 320 * 2 * 2                                              # 20 ms x 2 kanal x 2 bayt
        for i, k in enumerate(range(0, len(pcm), step)):
            s.sendto(struct.pack("!BBHII", 0x80, 97, i, i * 320, 7) + pcm[k:k + step], ("127.0.0.1", src.port))
            if i == 3:  # tuş sesi (DTMF) olayı: aynı akış, farklı yük türü
                s.sendto(struct.pack("!BBHII", 0x80, 101, 999, 0, 7) + b"\x01\x00\x00\xa0", ("127.0.0.1", src.port))
            time.sleep(0.001)

    threading.Thread(target=send).start()
    got = np.concatenate(list(src.frames()))
    assert abs(len(got) - 16000) < 400 and src.stats["yabanci"] == 1
    assert np.max(np.abs(got[1000:15000])) == pytest.approx(0.3, abs=0.02)   # iki kanalın ortalaması


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg yok")
def test_opus_sdp_ffmpeg_ile_alinir(tmp_path):
    enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if "libopus" not in enc:
        pytest.skip("ffmpeg'de libopus yok")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:   # boş port bul
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    text = f"v=0\nc=IN IP4 127.0.0.1\nm=audio {port} RTP/AVP 111\na=rtpmap:111 opus/48000/2\n"
    src = parse_source(StreamSpec(kaynak="", sdp=text, idle_timeout=1.5), False, tmp_path)
    wav = tmp_path / "t.wav"
    import soundfile as sf
    sf.write(wav, 0.3 * np.sin(2 * np.pi * 300 * np.arange(48000) / 16000), 16000)

    def send():
        time.sleep(1.0)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-re", "-i", str(wav), "-c:a", "libopus",
                        "-payload_type", "111", "-f", "rtp", f"rtp://127.0.0.1:{port}"], stdout=subprocess.DEVNULL)

    threading.Thread(target=send).start()
    got = np.concatenate(list(src.frames()))
    assert 2.5 < len(got) / 16000 < 3.5 and np.abs(got).max() > 0.1
    assert not list(tmp_path.glob("*.sdp"))                               # geçici SDP silindi
