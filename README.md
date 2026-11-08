# OpenRouter Free & Low-Cost Model Finder API (Rust)

Ultra-lightweight micro-service written in Rust (`axum` + `tokio`) to query and filter OpenRouter models based on real-time weekly popularity (`top-weekly`), pricing, and modalities.

## Kenapa service ini dibuat?
1. API official OpenRouter (`/api/v1/models`) tidak menyediakan filter sorting `top-weekly` maupun filter `output_modalities=text`.
2. Halaman web OpenRouter melakukan filtering secara client-side di browser.
3. Service ini menggabungkan stream RSC `top-weekly` OpenRouter dengan katalog resmi, di-cache dalam memori RAM, dan menyediakan REST API berkecepatan **< 1ms**.

---

## Menjalankan Server (Dev Mode)

```bash
cargo run
```
*Server otomatis berjalan di `http://127.0.0.1:3033` (atau port dari env variable `PORT`).*

---

## Endpoints

### 1. OpenAI-Compatible Endpoint: `GET /v1/models`
Format output 100% standar OpenAI (`{"object": "list", "data": [{"id": "...", "object": "model", ...}]}`).
Bisa langsung dihubungkan ke **Omniroute**, LiteLLM, OpenCode, LibreChat, dsb.

```bash
# Default: Mengembalikan model Free Text terurut Top-Weekly
curl "http://127.0.0.1:3033/v1/models"

# Batasi 5 model teratas:
curl "http://127.0.0.1:3033/v1/models?limit=5"
```

### 2. Custom Rich Filter Endpoint: `GET /models`
Mengembalikan metadata lengkap (pricing, context length, modalities, reasoning support, weekly rank).

#### Query Parameters:
| Parameter | Type | Default | Options / Keterangan |
| :--- | :--- | :--- | :--- |
| `variant` | string | `free` | `free`, `paid`, `all` |
| `output` | string | `text` | `text`, `image`, atau `all` |
| `sort` | string | `top-weekly`| `top-weekly`, `cheapest`, `context`, `newest` |
| `tools` | bool | `null` | `true` (hanya model yang support function calling) |
| `max_price` | float | `null` | Maksimal harga prompt per token |
| `limit` | int | `20` | Batas jumlah model yang dikembalikan |

---

## Contoh Penggunaan (cURL)

```bash
# Model Gratis, Output Text, Urut Top-Weekly (Format Lengkap)
curl "http://127.0.0.1:3033/models?variant=free&output=text&sort=top-weekly&limit=5"

# Model Gratis yang support Tools / Function Calling
curl "http://127.0.0.1:3033/models?variant=free&tools=true"

# Model Murah Berbayar (Cheapest) Output Text
curl "http://127.0.0.1:3033/models?variant=paid&output=text&sort=cheapest&limit=5"
```
