# 🚚 Delivery Delay Prediction — Dashboard & Deployment Framework

Framework produksi untuk model **prediksi keterlambatan pengiriman** (dikembangkan di
`Delivery_Logistics_Predictive_Analysis.ipynb`), dibungkus menjadi aplikasi yang siap
dipakai oleh tim operasional — lengkap dengan UI interaktif, containerization,
automated testing, dan CI/CD.

| Kebutuhan | Tools |
|---|---|
| UI / Dashboard | **Streamlit** |
| Deployment / Containerization | **Docker** + docker-compose |
| Testing | **pytest** |
| CI/CD otomatis | **GitHub Actions** |

---

## 1. Struktur Proyek

```
delivery-delay-app/
├── app/
│   ├── __init__.py
│   ├── model_utils.py       # load model, ambil opsi kategori, prediksi
│   └── streamlit_app.py     # UI dashboard Streamlit (entry point)
├── model/
│   ├── model.pkl             # pipeline sklearn (preprocessing + model) — taruh di sini
│   ├── config.json           # metadata & metrik model
│   └── sample_input.json     # contoh skema input
├── scripts/
│   ├── train_model.py        # replikasi pipeline notebook -> menghasilkan model.pkl
│   └── entrypoint.sh          # startup Docker: download model (jika perlu) + bind $PORT
├── tests/
│   ├── conftest.py           # fixture: dummy pipeline utk testing tanpa model asli
│   ├── test_model_utils.py   # unit test loading model & prediksi
│   └── test_app_logic.py     # unit test validasi input & format output UI
├── .github/workflows/
│   └── ci-cd.yml             # lint -> test -> build docker image -> (push)
├── Dockerfile
├── docker-compose.yml
├── railway.toml               # config healthcheck untuk deploy Railway
├── requirements.txt
├── .gitattributes             # Git LFS tracking utk model.pkl
├── .dockerignore
├── .gitignore
└── README.md
```

## 2. Cara Menjalankan (Lokal)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Siapkan model — pilih salah satu:
#    a) Sudah punya model.pkl dari notebook? -> copy ke folder model/
#    b) Belum? jalankan training ulang (butuh Delivery_Logistics.csv):
python scripts/train_model.py --data path/to/Delivery_Logistics.csv --out model/

# 2) Jalankan dashboard
streamlit run app/streamlit_app.py
```

Buka `http://localhost:8501` di browser.

## 3. Menjalankan Test

```bash
pytest -v --cov=app tests/
```

Test tidak butuh `model.pkl` asli — ada fixture pipeline dummy (`tests/conftest.py`)
sehingga CI tetap bisa jalan walau model belum di-training.

## 4. Menjalankan dengan Docker

```bash
docker build -t delivery-delay-app .
docker run -p 8501:8501 -v $(pwd)/model:/app/model delivery-delay-app

# atau
docker compose up --build
```

## 5. Menyimpan model.pkl di repo (Git LFS)

Supaya CI/CD bisa build image Docker yang sudah berisi model terlatih,
`model.pkl` perlu ikut ter-commit ke repo. Karena ukurannya bisa besar,
proyek ini pakai **Git LFS** (lihat `.gitattributes`) alih-alih commit
biasa:

```bash
git lfs install                     # sekali saja per komputer
git add model/model.pkl model/config.json model/sample_input.json .gitattributes
git commit -m "Add trained model artifact"
git push
```

Belum punya `git-lfs`? Install dulu: `brew install git-lfs` (Mac),
`sudo apt install git-lfs` (Ubuntu/Debian), atau lihat
[git-lfs.com](https://git-lfs.com) untuk platform lain.

> Alternatif: kalau tim sudah pakai S3/GCS untuk artefak ML, workflow CI/CD
> bisa diubah untuk `aws s3 cp` / `gsutil cp` model dari bucket alih-alih
> Git LFS — beri tahu saya kalau butuh versi itu.

## 6. Deploy ke Railway

Railway **tidak menarik file Git LFS** saat build (beda dengan GitHub Actions),
jadi model perlu di-download dari URL publik saat container start —
`scripts/entrypoint.sh` sudah menangani ini otomatis lewat env var `MODEL_URL`.

**Langkah-langkah:**

1. **Upload model.pkl ke GitHub Release** (supaya punya URL download publik):
   ```bash
   # tag rilis dulu
   git tag model-v1
   git push origin model-v1

   # lalu di GitHub: repo -> Releases -> Draft a new release
   # pilih tag "model-v1" -> upload model/model.pkl dan model/config.json
   # sebagai release assets -> Publish release
   ```
   Setelah publish, klik kanan pada file `model.pkl` di halaman release →
   copy link. Formatnya kira-kira:
   ```
   https://github.com/USERNAME/REPO/releases/download/model-v1/model.pkl
   ```

2. **Buat project di Railway:**
   - Buka [railway.app](https://railway.app) → login pakai GitHub.
   - **New Project → Deploy from GitHub repo** → pilih repo `delivery-delay-app`.
   - Railway otomatis mendeteksi `Dockerfile` dan `railway.toml`.

3. **Set environment variable** di Service → tab **Variables**:
   | Key | Value |
   |---|---|
   | `MODEL_URL` | link `model.pkl` dari langkah 1 |
   | `CONFIG_URL` | link `config.json` dari langkah 1 *(opsional)* |

   Kamu **tidak perlu** set `PORT` — Railway mengisinya otomatis, dan
   `scripts/entrypoint.sh` sudah membaca `$PORT` tersebut.

4. **Deploy** — Railway build otomatis setelah variable disimpan (atau klik
   **Deploy** manual). Pantau log build/deploy dari tab **Deployments**.

5. **Generate domain publik:** Settings → Networking → **Generate Domain**.
   Dashboard bisa diakses di `https://<nama-app>.up.railway.app`.

6. **Update model di masa depan:** upload `model.pkl` versi baru sebagai
   release baru (mis. tag `model-v2`), update value `MODEL_URL` di Railway
   Variables ke link yang baru, lalu redeploy (Railway → Deployments →
   **Redeploy**).

> Alternatif tanpa GitHub Release: host `model.pkl` di object storage apa pun
> yang punya direct-download URL (S3 public bucket, Cloudflare R2, Google
> Cloud Storage public URL, dll) — cukup isi `MODEL_URL` dengan link itu,
> mekanismenya sama persis.



## 7. CI/CD (GitHub Actions)

Workflow `.github/workflows/ci-cd.yml` berjalan otomatis pada setiap `push`/`pull_request`
ke branch `main`:

1. **Lint** — `flake8` untuk cek kualitas kode.
2. **Test** — `pytest` dengan coverage report (tidak butuh model.pkl asli).
3. **Build** — checkout dengan `lfs: true` (menarik `model.pkl`), lalu build image Docker.
4. **(Opsional) Push** — push image ke GitHub Container Registry saat merge ke `main`
   (butuh secret bawaan `GITHUB_TOKEN`, sudah otomatis tersedia).

Jika belum ada `model.pkl` yang ter-commit sama sekali (misal PR pertama),
job build tetap lolos dengan folder `model/` kosong — aplikasi akan
menampilkan pesan "model tidak ditemukan" yang ramah saat runtime,
bukan error/crash.

## 8. Catatan Penting

- Model **tidak boleh** menggunakan `delivery_time_hours`, `delivery_status`, atau
  `delivery_rating` sebagai fitur — kolom-kolom tersebut hanya tersedia *setelah*
  pengiriman selesai (data leakage), sesuai temuan di notebook asli.
- Fitur yang dipakai model: `distance_km`, `package_weight_kg`, `expected_time_hours`,
  `delivery_cost`, `delivery_partner`, `package_type`, `vehicle_type`, `delivery_mode`,
  `region`, `weather_condition`.
- Dashboard membaca daftar opsi dropdown **langsung dari `OneHotEncoder` di dalam
  `model.pkl`**, jadi tidak perlu hardcode kategori — otomatis sinkron dengan data
  training kapan pun model di-retrain.
