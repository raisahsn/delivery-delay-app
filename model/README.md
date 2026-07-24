# model/

Folder ini menyimpan artefak model hasil training (dari notebook atau
`scripts/train_model.py`). Isi folder ini **tidak** di-commit ke git secara
default (lihat `.gitignore`) karena `model.pkl` bisa berukuran besar dan
biasanya dianggap sebagai artefak build, bukan source code.

File yang seharusnya ada di sini setelah training:

| File | Deskripsi |
|---|---|
| `model.pkl` | Pipeline scikit-learn utuh (ColumnTransformer + classifier terbaik hasil tuning) |
| `config.json` | Metadata: tipe model, hyperparameter terbaik, metrik evaluasi |
| `sample_input.json` | Contoh skema input satu baris data pengiriman |

## Cara mengisi folder ini

**Opsi A — dari notebook:** copy `model.pkl`, `config.json`, dan
`sample_input.json` hasil Section 9 notebook ke folder ini.

**Opsi B — training ulang via script:**
```bash
python scripts/train_model.py --data Delivery_Logistics.csv --out model/
```

Setelah itu, jalankan dashboard dengan `streamlit run app/streamlit_app.py`
atau `docker compose up --build`.

## Meng-commit model.pkl ke repo (via Git LFS)

Folder ini sudah dikonfigurasi memakai **Git LFS** (lihat `.gitattributes` di
root proyek), jadi `model.pkl` bisa ikut di-push ke GitHub seperti file
biasa, dan otomatis ter-download saat CI/CD build image Docker.

```bash
# sekali saja per komputer:
git lfs install

# setelah model.pkl / config.json ada di folder ini:
git add model/model.pkl model/config.json model/sample_input.json .gitattributes
git commit -m "Add trained model artifact (via Git LFS)"
git push
```

Cek apakah file benar-benar masuk LFS (bukan ke history git biasa):
```bash
git lfs ls-files
# harus muncul: model/model.pkl
```
