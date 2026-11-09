# OpenRouter Free & Low-Cost Smart Proxy (Rust)

Ultra-lightweight micro-service written in Rust (`axum` + `tokio`) that acts as an OpenAI-compatible smart proxy and discovery engine for OpenRouter.

## Fitur Utama
1. **Dynamic Top-Weekly Ranking:** Mengurutkan model gratis berdasarkan token usage mingguan aktual OpenRouter (bukan daftar statis).
2. **OpenAI-Compatible Chat Completions:** Mendukung `POST /v1/chat/completions` dengan full streaming (SSE).
3. **Smart Virtual Models:**
   - `auto-free` / `free:top-1`: Otomatis memilih model gratis terbaik/terpopuler minggu ini.
   - `free:top-2`: Cadangan model gratis #2.
   - `free:top-3`: Cadangan model gratis #3.
   - `free:coding`: Model gratis spesialis coding (`cohere/north-mini-code:free`).
   - `free:context`: Model gratis dengan context window terbesar (1M tokens).
   - `free:tools`: Model gratis teratas yang mendukung function calling.
   - `cheapest:text`: Model berbayar paling murah meriah.
4. **Auto-Fallback Transparan:** Jika model gratis pilihan terkena rate-limit (`429`) atau down (`502`/`503`), proxy otomatis mencoba kandidat gratis berikutnya tanpa memutus chat.
5. **Key Passthrough:** Meneruskan header `Authorization: Bearer <key>` milik client/user secara langsung ke OpenRouter (dengan opsi fallback ke `.env` untuk pengujian lokal).
6. **In-Memory Refresh:** Sinkronisasi katalog berjalan di background setiap 15 menit, latency response API lokal **< 1ms**.
7. **Timeout per Attempt:** Setiap percobaan ke satu model dibatasi `UPSTREAM_TIMEOUT_SECS` (default `60`) lalu pindah ke kandidat berikutnya, jadi satu model hang tidak menahan request berjam-jam.
8. **Streaming Aman:** Byte stream ditahan sampai token teks pertama muncul. Stream yang hanya berisi reasoning (client akan melihatnya kosong) dianggap gagal dan di-fallback ke kandidat berikutnya.
9. **Modality & Context Aware:** Request bergambar/audio/file/video hanya diarahkan ke model yang mendukung modality tersebut, dan prompt besar difilter berdasarkan `context_length`.

---

## Cara Menjalankan

### Mode Dev
```bash
cargo run
```
*Server berjalan di `http://127.0.0.1:3033` (port dapat diubah via env `PORT`).*

### Environment Variables
| Env | Default | Fungsi |
| :--- | :--- | :--- |
| `PORT` | `3033` | Port HTTP |
| `HOST` | `0.0.0.0` | Bind address |
| `LAN_IP` | `localhost` | Hanya untuk log hint Base URL |
| `UPSTREAM_TIMEOUT_SECS` | `60` | Budget per percobaan ke satu model |
| `OPENROUTER_API_KEY` | – | Fallback key bila client tidak mengirim `Authorization` |


---

## Cara Menjalankan

### Mode Dev
```bash
cargo run
```
*Server berjalan di `http://127.0.0.1:3033` (port dapat diubah via env `PORT`).*

### Mode Production (Compile Binary)
```bash
cargo build --release
./target/release/openrouter-free-finder.exe
```

---

## Deploy di Debian (systemd user service)

Lokasi: `~/Apps/openrouter-free-finder` di `192.168.1.201`, port `3033`.

```bash
# build (pakai rustup toolchain, bukan cargo paket Debian yang masih 1.85)
cd ~/Apps/openrouter-free-finder
~/.cargo/bin/cargo build --release

# service
systemctl --user daemon-reload
systemctl --user enable --now openrouter-free-finder
systemctl --user status openrouter-free-finder

# log
journalctl --user -u openrouter-free-finder -f
```

Update kode setelah edit:
```bash
cd ~/Apps/openrouter-free-finder
~/.cargo/bin/cargo build --release && systemctl --user restart openrouter-free-finder
```

File service: `~/.config/systemd/user/openrouter-free-finder.service`
(`WorkingDirectory`, `EnvironmentFile=.env`, `LAN_IP`, `Restart=always`).
Linger sudah aktif, jadi service tetap jalan tanpa login.

---

## Setup di Omniroute / Client OpenAI

- **Base URL:** `http://192.168.1.201:3033/v1` (atau `http://host.docker.internal:3033/v1` bila client-nya container Docker)
- **API Key:** Masukkan OpenRouter API Key kamu sendiri.
- **Model Name:** Pilih `auto-free` (atau salah satu dari virtual models di atas).

### Kalau OmniRoute jalan di Docker
`127.0.0.1` di dalam container menunjuk ke container itu sendiri, bukan ke host.
Pilih salah satu:

1. Pakai IP LAN host: `http://192.168.1.201:3033/v1`
2. Pakai `host.docker.internal:3033/v1` — butuh satu baris di compose service:
   ```yaml
   extra_hosts:
     - "host.docker.internal:host-gateway"
   ```
   lalu `docker compose up -d omniroute`. Sudah terpasang di `~/Apps/OmniRoute/docker-compose.yml`.


---

## Endpoints

### 1. OpenAI Discovery: `GET /v1/models`
Mengembalikan daftar model gratis terurut top-weekly dalam format standar OpenAI.

```bash
curl http://127.0.0.1:3033/v1/models
```

### 2. OpenAI Chat Completion: `POST /v1/chat/completions`
Forwarding chat request ke OpenRouter dengan dukungan streaming SSE.

```bash
curl -X POST http://127.0.0.1:3033/v1/chat/completions \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto-free",
    "messages": [{"role": "user", "content": "Halo!"}]
  }'
```

### 3. Rich Catalog Filter: `GET /models`
Mendapatkan metadata lengkap model (pricing, context length, modalities, reasoning).

| Query Param | Tipe | Default | Nilai / Contoh |
| :--- | :--- | :--- | :--- |
| `variant` | string | `free` | `free`, `paid`, `all` |
| `output` | string | `text` | `text`, `all` |
| `sort` | string | `top-weekly`| `top-weekly`, `cheapest`, `context`, `newest` |
| `tools` | bool | `null` | `true` |
| `max_price` | float | `null` | `0.000001` |
| `limit` | int | `20` | `5` |
