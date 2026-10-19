/* Canlı altyazı istemcisi: ses yakalama + sunucu bağlantısı (WebSocket ya da WebRTC).
 *
 * Hem /canli sayfası hem ana sayfadaki "Mikrofon" kaynağı bunu kullanır.
 *
 *   const c = new LiveClient({
 *     params: () => ({ language: "vi", silence_ms: 700, ... }),   // /ws/live sorgu parametreleri
 *     onMessage: (m) => ...,          // sunucu iletileri: ready, partial, final, discard, stats, language, error, done
 *     onStatus: (text, state) => ..., // "live" | "ok" | undefined
 *     onLevel: (rms) => ...,          // ses seviyesi göstergesi
 *     onStopped: () => ...,           // oturum bitti ya da başlatılamadı (düğmeleri sıfırla)
 *     transport: () => "auto",       // "auto" | "webrtc" | "websocket"
 *     noiseSuppression: () => true,
 *   });
 *   await c.start({ type: "mic", deviceId }) | c.start({ type: "file", file });  c.stop();
 *
 * İletim yolları:
 *   WebSocket: 100 ms'lik 16 kHz Int16 paketler; kopunca ses biriktirilir, aynı oturuma dönülür.
 *              Aynı bilgisayarda en sağlam yol.
 *   WebRTC:    tarayıcı sesi Opus ile UDP'den yollar (paket kaybına dayanıklı, düşük gecikme);
 *              mesajlar "altyazi" veri kanalından. Başka cihazdan (telefon, Wi-Fi) bağlanırken.
 *   auto:      sayfa başka cihazdan açıldıysa ve sunucuda WebRTC varsa WebRTC, yoksa WebSocket;
 *              WebRTC kurulamazsa kendiliğinden WebSocket'e döner.
 */
(function () {
  const WORKLET = `
class Pcm16 extends AudioWorkletProcessor {
  constructor() { super(); this.ratio = sampleRate / 16000; this.pos = 0; this.buf = new Float32Array(1600); this.n = 0; this.sq = 0; this.cnt = 0; }
  process(inputs) {
    const ch = inputs[0][0];
    if (!ch) return true;
    for (let k = 0; k < ch.length; k++) { this.sq += ch[k] * ch[k]; }
    this.cnt += ch.length;
    while (this.pos < ch.length) {
      const i = this.pos | 0, f = this.pos - i;
      const s = ch[i] * (1 - f) + (i + 1 < ch.length ? ch[i + 1] : ch[i]) * f;
      this.buf[this.n++] = s;
      if (this.n === this.buf.length) {
        const out = new Int16Array(this.n);
        for (let j = 0; j < this.n; j++) { const v = Math.max(-1, Math.min(1, this.buf[j])); out[j] = v * 32767; }
        this.port.postMessage({ pcm: out.buffer, rms: Math.sqrt(this.sq / this.cnt) }, [out.buffer]);
        this.n = 0; this.sq = 0; this.cnt = 0;
      }
      this.pos += this.ratio;
    }
    this.pos -= ch.length;
    return true;
  }
}
registerProcessor("pcm16", Pcm16);`;

  const LOCAL = ["localhost", "127.0.0.1", "[::1]"];
  const PENDING_MAX = 600; // bağlantı yokken en fazla 60 sn ses biriktir (WebSocket)

  function secureHint() {
    return window.isSecureContext ? "" :
      " Başka cihazdan mikrofon için sayfa https:// ile açılmalı (sunucuyu HTTPS=1 ile başlatın).";
  }

  class LiveClient {
    constructor(opts) {
      this.o = opts;
      this.running = false;
      this.offset = 0;          // yeni oturum açılınca ekrandaki zamanın kayması (sn)
      this.mode = null;         // "websocket" | "webrtc" (bu oturumda kullanılan)
      this.sessionId = null; this.resumeToken = null; this.lastId = -1;
      this._reset();
    }

    _reset() {
      this.ws = null; this.pc = null; this.dc = null;
      this.ctx = null; this.node = null; this.src = null; this.mic = null; this.sendTrack = null;
      this.gotReady = false; this.fatal = false; this.captured = 0; this.pending = [];
      clearTimeout(this.retryTimer); this.retryTimer = null;
    }

    status(text, state) { this.o.onStatus?.(text, state); }

    /* ------------------------------------------------------------ ses */
    async _context(preferred16k) {
      const c = preferred16k ? new AudioContext({ sampleRate: 16000 }) : new AudioContext();
      await c.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" })));
      return c;
    }

    async _openAudio(kind) {
      if (kind.type === "mic") {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error("Tarayıcı mikrofona izin vermiyor." + secureHint());
        const ns = this.o.noiseSuppression !== undefined ? (typeof this.o.noiseSuppression === "function" ? this.o.noiseSuppression() : this.o.noiseSuppression) : true;
        const agc = this.o.autoGainControl !== undefined ? (typeof this.o.autoGainControl === "function" ? this.o.autoGainControl() : this.o.autoGainControl) : true;
        const ec = this.o.echoCancellation !== undefined ? (typeof this.o.echoCancellation === "function" ? this.o.echoCancellation() : this.o.echoCancellation) : false;
        this.mic = await navigator.mediaDevices.getUserMedia({ audio: {
          deviceId: kind.deviceId ? { exact: kind.deviceId } : undefined,
          channelCount: 1,
          echoCancellation: Boolean(ec),
          noiseSuppression: Boolean(ns),
          autoGainControl: Boolean(agc) } });
        try { this.ctx = await this._context(true); this.src = this.ctx.createMediaStreamSource(this.mic); }
        catch { await this.ctx?.close().catch(() => {}); this.ctx = await this._context(false); this.src = this.ctx.createMediaStreamSource(this.mic); }
        this.sendTrack = this.mic.getAudioTracks()[0];
      } else {
        this.ctx = await this._context(false);
        const buf = await this.ctx.decodeAudioData(await kind.file.arrayBuffer());
        this.src = this.ctx.createBufferSource();
        this.src.buffer = buf;
        this.src.connect(this.ctx.destination); // dinleyerek takip et
        const dest = this.ctx.createMediaStreamDestination(); // WebRTC için dosyadan ses izi
        this.src.connect(dest);
        this.sendTrack = dest.stream.getAudioTracks()[0];
        this.src.onended = () => { if (this.running) setTimeout(() => this.stop(), 300); };
      }
      this.node = new AudioWorkletNode(this.ctx, "pcm16");
      this.node.port.onmessage = ({ data }) => {
        this.o.onLevel?.(data.rms);
        this.captured += data.pcm.byteLength / 2;
        if (this.mode !== "websocket") return; // WebRTC'de ses Opus iziyle gider
        if (this.ws && this.ws.readyState === 1 && this.gotReady) this.ws.send(data.pcm);
        else if (this.running) { this.pending.push(data.pcm); if (this.pending.length > PENDING_MAX) this.pending.shift(); }
      };
      this.src.connect(this.node);
    }

    _closeAudio() {
      try { this.src?.disconnect(); this.src?.stop?.(); } catch {}
      this.mic?.getTracks().forEach((t) => t.stop());
      this.node?.port.close();
      this.ctx?.close().catch(() => {});
      this.ctx = this.node = this.src = this.mic = this.sendTrack = null;
      this.o.onLevel?.(0);
    }

    /* ------------------------------------------------------------ ortak mesaj işleme */
    _query() {
      const p = Object.fromEntries(Object.entries(this.o.params()).filter(([, v]) => v !== undefined && v !== null && v !== "")
        .map(([k, v]) => [k, String(v)]));
      if (this.sessionId) { p.resume = this.sessionId; p.token = this.resumeToken; p.after = String(this.lastId); }
      return p;
    }

    _handle(m) {
      if (m.type === "ready") {
        if (!m.resumed) {
          // yeni oturum: sunucunun zamanı sıfırdan başlar; önceki oturumların sesi kadar kaydır
          // (ilk oturumda 0: WebRTC'de bağlantı kurulurken çekilen ses sunucuya gitmez)
          this.offset = this.sessionId ? this.captured / 16000 : 0; this.lastId = -1;
          if (this.sessionId) this.status("Önceki oturum kapanmıştı, yeni oturum başladı");
        }
        this.sessionId = m.session_id; this.resumeToken = m.resume_token; this.gotReady = true;
        if (this.mode === "websocket") while (this.pending.length) this.ws.send(this.pending.shift());
        this.status((m.resumed ? "Dinleniyor (bağlantı geri geldi)" : "Dinleniyor") + (this.mode === "webrtc" ? " · WebRTC" : ""), "live");
      } else if (m.type === "final") {
        if (m.id <= this.lastId) return; // geri bağlanınca yeniden gelen satır zaten ekranda
        this.lastId = m.id;
      } else if (m.type === "error") {
        if (!this.gotReady) this.fatal = true; // bağlantı reddedildi (anahtar, ayar): yeniden denemek boşuna
        this.status("Hata: " + m.message);
      } else if (m.type === "done") {
        this.sessionId = null;
        this.status(m.webhook ? "Tamamlandı · bildirim gönderildi" : "Tamamlandı", "ok");
        this._closeTransport();
        this.o.onStopped?.();
      }
      this.o.onMessage?.(m);
    }

    _lost() {
      // Bağlantı beklenmedik koptu
      this.gotReady = false;
      if (!this.running) return;
      if (this.fatal) { this._end(); return; }
      this.status("Bağlantı koptu, yeniden bağlanılıyor…" + (this.mode === "websocket" ? " (ses biriktiriliyor)" : ""));
      this.retryTimer = setTimeout(() => this._connect().catch((e) => this._failed(e)), 1500);
    }

    _end() {
      this.running = false; this._closeAudio(); this._closeTransport(); this.pending.length = 0;
      this.o.onStopped?.();
    }

    _failed(e) {
      this.status("Başlatılamadı: " + (e?.message || e));
      this._end();
    }

    _closeTransport() {
      try { this.ws?.close(); } catch {}
      try { this.dc?.close(); } catch {}
      try { this.pc?.close(); } catch {}
      this.ws = this.pc = this.dc = null;
    }

    /* ------------------------------------------------------------ WebSocket */
    _connectWs() {
      return new Promise((resolve) => {
        this.gotReady = false; this.fatal = false;
        const proto = location.protocol === "https:" ? "wss" : "ws";
        const ws = this.ws = new WebSocket(`${proto}://${location.host}/ws/live?${new URLSearchParams(this._query())}`);
        ws.binaryType = "arraybuffer";
        ws.onmessage = (e) => { this._handle(JSON.parse(e.data)); if (this.gotReady) resolve(); };
        ws.onclose = () => { if (this.ws === ws) { this.ws = null; this._lost(); } resolve(); };
      });
    }

    /* ------------------------------------------------------------ WebRTC */
    async _connectRtc() {
      this.gotReady = false; this.fatal = false;
      const pc = this.pc = new RTCPeerConnection({ iceServers: [] }); // kapalı ağ: STUN yok
      const dc = this.dc = pc.createDataChannel("altyazi", { ordered: true });
      pc.addTrack(this.sendTrack, new MediaStream([this.sendTrack]));
      const ready = new Promise((resolve, reject) => {
        const t = setTimeout(() => reject(new Error("WebRTC bağlantısı 15 sn'de kurulamadı")), 15000);
        dc.onmessage = (e) => { this._handle(JSON.parse(e.data)); if (this.gotReady) { clearTimeout(t); resolve(); } };
        dc.onclose = () => { clearTimeout(t); if (this.dc === dc) { this.dc = null; try { pc.close(); } catch {} this.pc = null; this._lost(); } reject(new Error("veri kanalı kapandı")); };
      });
      ready.catch(() => {});
      pc.onconnectionstatechange = () => { if (pc.connectionState === "failed" && this.pc === pc) dc.close(); };
      await pc.setLocalDescription(await pc.createOffer());
      await new Promise((res) => { // aynı ağdaki adaylar hızlı toplanır; en fazla 2 sn bekle
        if (pc.iceGatheringState === "complete") return res();
        const t = setTimeout(res, 2000);
        pc.addEventListener("icegatheringstatechange", () => { if (pc.iceGatheringState === "complete") { clearTimeout(t); res(); } });
      });
      const key = this.o.params().key;
      const r = await fetch("/webrtc/teklif", { method: "POST",
        headers: { "Content-Type": "application/json", ...(key ? { "X-API-Key": key } : {}) },
        body: JSON.stringify({ sdp: pc.localDescription.sdp, type: "offer", ayarlar: this._query() }) });
      if (!r.ok) {
        let msg = r.statusText; try { msg = (await r.json()).detail || msg; } catch {}
        const err = new Error(msg); err.status = r.status; throw err;
      }
      await pc.setRemoteDescription(await r.json());
      await ready;
    }

    /* ------------------------------------------------------------ yol seçimi */
    async _pickMode() {
      const want = this.o.transport ? this.o.transport() : "auto";
      if (want === "websocket" || !window.RTCPeerConnection) return "websocket";
      let server = false;
      try { server = !!(await (await fetch("/live/health")).json()).webrtc; } catch {}
      if (!server) {
        if (want === "webrtc") this.status("Sunucuda WebRTC yok (aiortc kurulu değil), WebSocket kullanılıyor");
        return "websocket";
      }
      if (want === "webrtc") return "webrtc";
      return LOCAL.includes(location.hostname) ? "websocket" : "webrtc";
    }

    async _connect() {
      if (!this.running) return;
      if (this.mode === "webrtc") {
        try { await this._connectRtc(); return; }
        catch (e) {
          this._closeTransport();
          if (!this.running) return;
          // anahtar / ayar hatası: WebSocket de aynı yanıtı alır
          if (e.status === 400 || e.status === 401 || this.fatal) throw e;
          if (this.gotReady) return; // oturum açıldı, sonra koptu: _lost yeniden dener
          console.warn("WebRTC kurulamadı, WebSocket'e dönülüyor:", e);
          this.status("WebRTC kurulamadı, WebSocket ile devam…");
          this.mode = "websocket";
        }
      }
      await this._connectWs();
    }

    async start(kind) {
      if (this.running) return;
      this._reset();
      this.running = true; this.offset = 0; this.sessionId = null; this.resumeToken = null; this.lastId = -1;
      this.status("Model hazırlanıyor…");
      try {
        this.mode = await this._pickMode();
        // WebRTC'de ses izi teklifte olmalı: ses önce açılır. WebSocket'te ses, oturum hazır olunca.
        if (this.mode === "webrtc") await this._openAudio(kind);
        await this._connect();
        if (!this.ctx && this.running) await this._openAudio(kind);
        if (this.src?.start) this.src.start();
      } catch (e) {
        this._failed(e);
      }
    }

    stop() {
      if (!this.running) return;
      this.running = false;
      clearTimeout(this.retryTimer);
      this._closeAudio();
      const msg = JSON.stringify({ type: "stop" });
      if (this.ws && this.ws.readyState === 1) { this.ws.send(msg); this.status("Son cümle işleniyor…"); }
      else if (this.dc && this.dc.readyState === "open") { this.dc.send(msg); this.status("Son cümle işleniyor…"); }
      else { this.status("Durduruldu"); this._closeTransport(); this.o.onStopped?.(); }
    }
  }

  window.LiveClient = LiveClient;
})();
