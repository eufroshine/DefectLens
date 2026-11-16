# DefectLens

Sistem analisis risiko *software defect* pada source code Java. DefectLens memberi skor risiko per file berdasarkan metrik ukuran dan kompleksitas kode serta riwayat perubahan Git, menjelaskan alasan skornya dengan SHAP, dan menampilkan temuan PMD berdampingan.

Proyek ini adalah bagian dari penelitian skripsi (Software Engineering). **Statusnya prototipe riset, bukan alat produksi.** Lihat bagian [Hasil awal dan keterbatasan](#hasil-awal-dan-keterbatasan) sebelum memakai skornya.

## Isi repository

| Bagian | Status | Keterangan |
| --- | --- | --- |
| `src/extract/` | Selesai | Ekstraksi fitur (Git, Lizard, PMD) dan penggabungan dengan label PROMISE |
| `src/model/` | Selesai | Eksperimen model, evaluasi kronologis, SHAP, dan analisis prioritas pemeriksaan |
| `dashboard/` | Prototipe | Dashboard Streamlit, termasuk analisis repository baru |
| `api/` (FastAPI) | Direncanakan | Layanan analisis untuk dashboard web |
| `web/` (Next.js + React) | Direncanakan | Dashboard web final |

## Cara kerja singkat

1. **Data dan label** diambil dari dataset PROMISE (proyek Apache ant, camel, lucene, poi; 16 rilis).
2. **Fitur** dihitung per file: ukuran dan kompleksitas (Lizard) serta riwayat perubahan 365 hari sebelum rilis (PyDriller): frekuensi perubahan, churn, jumlah penulis, kebaruan, dan jumlah commit perbaikan.
3. **Model** Random Forest (pembanding: XGBoost dan Logistic Regression) dievaluasi dengan validasi kronologis: latih pada rilis lama, uji pada rilis berikutnya.
4. **Penjelasan** memakai SHAP. Temuan **PMD** hanya ditampilkan sebagai informasi pendamping, tidak dipakai sebagai fitur.

## Prasyarat

- Python 3.13 (versi lain kemungkinan bisa, tetapi belum diuji)
- Git
- JDK 17 atau lebih baru (diuji dengan Eclipse Temurin 25)
- [PMD](https://github.com/pmd/pmd/releases) (diuji dengan 7.29.0)
- Windows (perintah di bawah memakai CMD; di Linux atau macOS gunakan `source .venv/bin/activate`)

## Instalasi

```bash
git clone https://github.com/eufroshine/DefectLens
cd defectlens
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Atur lokasi PMD. Untuk dashboard, pakai variabel lingkungan `PMD_PATH`; untuk `src/extract/run_pmd.py`, ubah konstanta `PMD` di bagian atas file.

```bash
set PMD_PATH=D:\Tools\pmd-bin-7.29.0\bin\pmd.bat
```

## Menjalankan pipeline penelitian

Folder `data/raw/` tidak disertakan di repository. Siapkan datanya dahulu:

1. Unduh dataset PROMISE (salinan `hjx-zhans/promise-backup` di GitHub) dan ekstrak ke `data/raw/PROMISE-backup-master/`.
2. Jalankan skrip berikut dari folder utama proyek, berurutan.

```bash
# 1. Clone repository proyek, hitung fitur riwayat Git kumulatif, gabungkan dengan PROMISE (merged_*.csv)
python src/extract/run_all.py

# 2. Fitur riwayat Git berbasis jendela 365 hari (merged2_*.csv)
python src/extract/build_period_features.py

# 3. Temuan PMD per file (merged3_*.csv), satu proyek per perintah
python src/extract/run_pmd.py ant

# 4. Fitur Lizard yang bisa dihitung untuk repository baru (merged4_*.csv)
python src/extract/build_standalone.py

# 5. Evaluasi dan penjelasan
python src/model/rolling_eval2.py      # skenario A, B, C
python src/model/rolling_eval3.py      # skenario A, C, D (+ PMD)
python src/model/rolling_eval4.py      # skenario C, E, L (fitur mandiri)
python src/model/ablation_priority.py  # recall pada 10% dan 20% file teratas
python src/model/explain_shap.py       # SHAP dan model final
```

Hasil tersimpan di `data/processed/` dan `models/`.

File lain di `src/model/` (`baseline.py`, `compare_features.py`, `compare_all.py`, `rolling_eval.py`) adalah eksperimen tahap awal yang dipertahankan sebagai catatan.

## Menjalankan dashboard prototipe

```bash
streamlit run dashboard/app.py
```

Jalankan dari folder utama proyek. Halaman yang tersedia:

- **Repository baru**: masukkan URL `https://...` atau folder lokal berisi repository Java. Sistem menghitung fitur, memberi skor risiko, dan menampilkan alasan serta temuan PMD per file.
- **Analisis file**: hasil untuk empat proyek PROMISE, lengkap dengan label bug sebenarnya.
- **Hasil evaluasi** dan **Tentang**.

## Hasil awal dan keterbatasan

Ringkasan studi pendahuluan (16 rilis, 12 pasangan latih-uji, selisih PR-AUC rata-rata):

| Perbandingan | Selisih | Catatan |
| --- | --- | --- |
| Metrik PROMISE + riwayat Git vs metrik PROMISE saja | +0,022 | Kecil dan tidak merata antarproyek |
| Menambah temuan PMD sebagai fitur | -0,002 | Tidak membantu |
| Fitur mandiri (Lizard + Git) vs metrik PROMISE + Git | -0,012 | Tidak berbeda bermakna |
| Riwayat Git di atas Lizard saja | +0,033 | Riwayat tetap menambah nilai |

Hal yang perlu diperhatikan:

- Model sekitar dua kali lebih efisien daripada pengurutan acak, tetapi **belum terbukti lebih baik daripada pengurutan sederhana berdasarkan ukuran file**.
- Nilai p yang dilaporkan hanya petunjuk kasar, karena pasangan latih-uji dalam satu proyek saling tumpang tindih.
- Skor risiko adalah keluaran model, **bukan probabilitas terkalibrasi**. Repository baru tidak punya label bug, sehingga benar atau salahnya skor tidak bisa diukur. Pakai urutannya, bukan angkanya.
- Temuan PMD adalah indikasi (code smell atau pelanggaran aturan), bukan bukti bug, dan jumlahnya berkorelasi dengan ukuran file.
- Label berasal dari PROMISE. File tanpa laporan bug belum tentu bebas bug.
- Model dilatih dari empat proyek Apache yang berumur lama. Hasilnya pada repository kecil atau baru (riwayat commit pendek) belum teruji.
- Hanya mendukung Java.

## Rencana

- [ ] FastAPI: endpoint analisis asinkron (`POST /api/analyses`, status, daftar file, detail file)
- [ ] Dashboard Next.js + React: input repository, status analisis, tabel file berperingkat, detail file (grafik SHAP dan temuan PMD), riwayat, evaluasi, dan halaman keterbatasan
- [ ] Pengujian (pytest untuk API, pengujian black-box untuk dashboard)
- [ ] Perluasan data (proyek PROMISE lain)

## Sumber data dan alat

- Dataset PROMISE: Sayyad Shirabad & Menzies (2005); data defect Java oleh Jureczko & Madeyski (2010).
- Proyek Apache (ant, camel, lucene, poi) mengikuti lisensi masing-masing.
- Alat: [PyDriller](https://github.com/ishepard/pydriller), [Lizard](https://github.com/terryyin/lizard), [PMD](https://pmd.github.io), [scikit-learn](https://scikit-learn.org), [XGBoost](https://xgboost.ai), [SHAP](https://github.com/shap/shap), [Streamlit](https://streamlit.io).

## License

This project is licensed under the [MIT License](LICENSE).

See the [LICENSE](LICENSE) file for details.
