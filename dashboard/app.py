"""DefectLens: dashboard hasil analisis risiko defect dan temuan PMD.

Jalankan dari folder utama proyek (D:\\Web\\defectlens):
    streamlit run dashboard\\app.py
"""
import os

import pandas as pd
import streamlit as st

import core

st.set_page_config(page_title="DefectLens", layout="wide")

DATA = core.DATA


@st.cache_data(show_spinner=False)
def cached_versions():
    return core.list_versions()


@st.cache_resource(show_spinner="Melatih model dari rilis sebelumnya...")
def cached_analysis(project, version):
    return core.analyze(project, version)


@st.cache_resource(show_spinner="Melatih model dari data ant, poi, lucene, camel...")
def cached_model():
    import newrepo
    return newrepo.train_model()


def page_analysis():
    versions = cached_versions()
    if not versions:
        st.error("Tidak ada file merged3_*.csv di data/processed. "
                 "Jalankan dulu src\\extract\\run_pmd.py, dan pastikan "
                 "aplikasi dijalankan dari folder utama proyek.")
        return

    st.title("Analisis risiko per file")
    st.caption("Model dilatih hanya dari rilis sebelumnya, lalu menilai rilis "
               "yang dipilih. Model tidak pernah melihat rilis yang dinilai.")

    c1, c2, c3 = st.columns(3)
    project = c1.selectbox("Proyek", list(versions))
    options = versions[project][1:]          # versi pertama tidak punya data latih
    if not options:
        st.warning("Proyek ini hanya punya satu versi, jadi belum ada data latih.")
        return
    version = c2.selectbox("Rilis yang dinilai", options, index=len(options) - 1)
    order = c3.radio("Urutkan berdasarkan",
                     ["Risiko model", "Skor gabungan (model + PMD)"],
                     horizontal=True)

    res = cached_analysis(project, version)
    if res is None:
        st.warning("Data latih untuk rilis ini tidak punya kelas bug, "
                   "jadi model tidak bisa dilatih.")
        return

    table = core.build_table(res)
    key = "risiko_ml" if order == "Risiko model" else "skor_gabungan"
    table = table.sort_values(key, ascending=False).reset_index(drop=True)

    show_truth = st.checkbox("Tampilkan label bug sebenarnya (dari PROMISE)",
                             value=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("File dinilai", len(table))
    m2.metric("Data latih", res["n_train"])
    m3.metric("Risiko ≥ 0,5", int((table["risiko_ml"] >= 0.5).sum()))
    if show_truth and table["bug_asli"].sum() > 0:
        r20 = core.recall_at(table[key].values, table["bug_asli"].values, 0.2)
        m4.metric("Bug ditemukan di 20% teratas", f"{r20:.0%}")
    else:
        m4.metric("Bug ditemukan di 20% teratas", "-")

    st.caption("Angka risiko adalah skor model, bukan probabilitas yang "
               "terkalibrasi. Temuan PMD adalah indikasi (code smell atau "
               "pelanggaran aturan), bukan bukti adanya bug.")

    top_n = st.slider("Jumlah file ditampilkan", 10, min(200, len(table)),
                      min(30, len(table)))
    view = table.head(top_n).copy()
    cols = ["kelas", "risiko_ml", "pmd_total", "pmd_high", "alasan"]
    if order != "Risiko model":
        cols.insert(2, "skor_gabungan")
    if show_truth:
        view["bug_asli"] = view["bug_asli"].map({1: "ya", 0: "tidak"})
        cols.append("bug_asli")
    st.dataframe(
        view[cols], hide_index=True,
        column_config={
            "kelas": "Kelas",
            "risiko_ml": st.column_config.ProgressColumn(
                "Risiko model", min_value=0.0, max_value=1.0, format="%.2f"),
            "skor_gabungan": st.column_config.ProgressColumn(
                "Skor gabungan", min_value=0.0, max_value=1.0, format="%.2f"),
            "pmd_total": "Temuan PMD",
            "pmd_high": "PMD prioritas tinggi",
            "alasan": "Tiga alasan utama",
            "bug_asli": "Bug sebenarnya",
        })

    st.subheader("Detail satu file")
    chosen = st.selectbox("Pilih kelas", view["kelas"].tolist())
    row = table[table["kelas"] == chosen].iloc[0]
    i = int(res["raw"].index[res["raw"]["name"] == chosen][0])

    d1, d2, d3 = st.columns(3)
    d1.metric("Risiko model", f"{row['risiko_ml']:.2f}")
    d2.metric("Temuan PMD", int(row["pmd_total"]))
    d3.metric("PMD prioritas tinggi", int(row["pmd_high"]))
    st.caption(f"File: {row['file']}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Kenapa file ini berisiko**")
        st.caption("Batang positif menaikkan risiko, batang negatif "
                   "menurunkannya (kontribusi SHAP).")
        contrib = core.contributions(res, i, top=10)
        st.bar_chart(contrib.set_index("fitur")["kontribusi"])
        st.dataframe(contrib, hide_index=True, column_config={
            "fitur": "Fitur", "nilai_asli": "Nilai", "kontribusi": "Kontribusi"})
    with right:
        st.markdown("**Temuan PMD pada file ini**")
        found = core.pmd_findings(project, version, row["file"])
        if found is None:
            st.info("File hasil mentah PMD tidak ditemukan "
                    f"(pmd_raw_{project}-{version}.csv).")
        elif found.empty:
            st.success("PMD tidak menemukan apa pun pada file ini. Bisa juga "
                       "berarti PMD gagal membaca kodenya.")
        else:
            st.dataframe(found, hide_index=True)


def page_newrepo():
    st.title("Analisis repository baru")
    st.caption("Masukkan repository Java. Sistem menghitung ukuran dan "
               "kompleksitas kode (lizard), riwayat perubahan 365 hari terakhir "
               "(Git), dan temuan PMD, lalu mengurutkan file berdasarkan skor "
               "risiko.")

    source = st.text_input(
        "URL repository atau folder lokal",
        placeholder="https://github.com/pengguna/proyek.git  atau  D:\\proyek\\repo")
    if st.button("Analisis", type="primary", disabled=not source.strip()):
        try:
            import newrepo
        except ImportError as e:
            st.error(f"Library belum terpasang: {e}. Jalankan "
                     "pip install lizard pydriller shap")
            return
        try:
            model, n_train = cached_model()
        except Exception as e:
            st.error(str(e))
            return
        with st.status("Menganalisis repository...", expanded=True) as status:
            try:
                res = newrepo.analyze_repo(source, model, log=st.write)
            except Exception as e:
                status.update(label="Analisis gagal", state="error")
                st.error(str(e))
                return
            status.update(label="Analisis selesai", state="complete")
        st.session_state["newrepo_res"] = res

    res = st.session_state.get("newrepo_res")
    if res is None:
        st.info("Masukkan URL atau folder repository, lalu tekan Analisis. "
                "Repository besar bisa memerlukan beberapa menit.")
        return

    import newrepo
    m1, m2, m3 = st.columns(3)
    m1.metric("File Java dianalisis", res["n_files"])
    m2.metric("Commit dalam 365 hari", res["commits"])
    m3.metric("Commit terakhir", res["anchor"].strftime("%Y-%m-%d"))
    for w in res["warnings"]:
        st.warning(w)
    st.caption("Gunakan urutannya, bukan angkanya: skor adalah keluaran model, "
               "bukan probabilitas yang terkalibrasi, dan model dilatih dari "
               "empat proyek Apache yang lama. Repository ini tidak punya label "
               "bug, jadi benar atau salahnya skor tidak bisa diukur. Temuan "
               "PMD adalah indikasi, bukan bukti bug.")

    table = res["table"]
    if len(table) > 5:
        top_n = st.slider("Jumlah file ditampilkan", 5, min(200, len(table)),
                          min(30, len(table)), key="nr_top")
    else:
        top_n = len(table)
    view = table.head(top_n)
    st.dataframe(
        view[["file", "persentil", "risiko_ml", "pmd_total", "pmd_high", "alasan"]],
        hide_index=True,
        column_config={
            "file": "File",
            "persentil": st.column_config.ProgressColumn(
                "Peringkat risiko", min_value=0.0, max_value=1.0, format="%.2f"),
            "risiko_ml": st.column_config.NumberColumn("Skor model", format="%.2f"),
            "pmd_total": "Temuan PMD",
            "pmd_high": "PMD prioritas tinggi",
            "alasan": "Tiga alasan utama",
        })

    st.subheader("Detail satu file")
    chosen = st.selectbox("Pilih file", view["file"].tolist(), key="nr_file")
    row = table[table["file"] == chosen].iloc[0]
    left, right = st.columns(2)
    with left:
        st.markdown("**Kenapa file ini berisiko**")
        st.caption("Batang positif menaikkan risiko, batang negatif "
                   "menurunkannya (kontribusi SHAP).")
        contrib = newrepo.contributions(res, int(row["idx"]), top=10)
        st.bar_chart(contrib.set_index("fitur")["kontribusi"])
        st.dataframe(contrib, hide_index=True, column_config={
            "fitur": "Fitur", "nilai_asli": "Nilai", "kontribusi": "Kontribusi"})
    with right:
        st.markdown("**Temuan PMD pada file ini**")
        found = newrepo.pmd_for_file(res, chosen)
        if found.empty:
            st.success("PMD tidak menemukan apa pun pada file ini.")
        else:
            st.dataframe(found, hide_index=True)


def page_results():
    st.title("Hasil evaluasi")
    st.caption("Ringkasan percobaan: latih di rilis lama, uji di rilis berikutnya. "
               "Nilai p dari semua percobaan hanya perkiraan kasar karena pasangan "
               "latih-uji saling tumpang tindih.")

    p = f"{DATA}/rolling_eval3.csv"
    if os.path.exists(p):
        r = pd.read_csv(p)
        st.subheader("PR-AUC menurut kelompok fitur")
        st.caption("A = metrik PROMISE saja, C = A + riwayat Git (1 tahun), "
                   "D = C + temuan PMD.")
        st.dataframe(r.groupby(["model", "fitur"])["pr_auc"].mean()
                     .unstack().round(3))
    else:
        st.info("rolling_eval3.csv belum ada. Jalankan src\\model\\rolling_eval3.py.")

    p = f"{DATA}/ablation_priority.csv"
    if os.path.exists(p):
        a = pd.read_csv(p)
        st.subheader("Prioritas pemeriksaan")
        st.caption("Persentase file bermasalah yang ditemukan bila hanya "
                   "memeriksa file teratas. Pembanding acak: 10% dan 20%.")
        t = a.pivot_table(index="metode", columns="top", values="recall").round(3)
        t.columns = [f"{int(c * 100)}% file teratas" for c in t.columns]
        st.dataframe(t)
    else:
        st.info("ablation_priority.csv belum ada.")

    p = f"{DATA}/shap_importance.csv"
    if os.path.exists(p):
        s = pd.read_csv(p)
        st.subheader("Fitur paling berpengaruh (SHAP)")
        st.bar_chart(s.head(15).set_index("fitur")["mean_abs_shap"])
    else:
        st.info("shap_importance.csv belum ada.")


def page_about():
    st.title("Tentang DefectLens")
    st.markdown(
        """
DefectLens menilai file Java mana yang lebih mungkin bermasalah, lalu
menampilkan alasannya dan temuan alat pemeriksa kode (PMD) berdampingan.

**Data**
- Label dan metrik kode: dataset PROMISE (ant, camel, lucene, poi).
- Riwayat perubahan: repository Git proyek yang sama, jendela 365 hari
  sebelum tanggal rilis.
- Temuan static analysis: PMD, aturan quickstart.

**Keterbatasan yang perlu diingat**
- Halaman "Analisis file" menampilkan hasil untuk empat proyek yang sudah
  diolah (ada kunci jawaban dari PROMISE). Halaman "Repository baru" menilai
  repository Java apa pun, tetapi tanpa kunci jawaban.
- Model untuk repository baru dilatih dari empat proyek Apache yang lama dan
  hanya memakai fitur yang bisa dihitung sendiri (lizard dan riwayat Git).
- Label bug berasal dari PROMISE. File tanpa laporan bug belum tentu bebas bug.
- Pada data ini, model belum terbukti mengungguli pengurutan sederhana
  berdasarkan ukuran file.
- Temuan PMD sebagian besar berkorelasi dengan ukuran file, dan menambahkannya
  sebagai fitur tidak memperbaiki prediksi.
- Kode versi lama yang tidak terbaca PMD tercatat tanpa temuan.
"""
    )


page = st.sidebar.radio("Halaman", ["Repository baru", "Analisis file",
                                    "Hasil evaluasi", "Tentang"])
if page == "Repository baru":
    page_newrepo()
elif page == "Analisis file":
    page_analysis()
elif page == "Hasil evaluasi":
    page_results()
else:
    page_about()