from pathlib import Path

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent


def render() -> None:
    st.html(BASE_DIR / "Assets" / "about.css")
    st.html("""
        <section class="about-page" aria-labelledby="about-title">
            <h1 id="about-title" class="about-title">About</h1>
            <div class="about-description">
                <p>Aplikasi prediksi perubahan harga Newcastle Coal Futures pada periode waktu 1 hari, 7 hari, dan 14 hari kedepan 
                berdasarkan harga WTI Crude Oil, Harga Henry Hub Natural Gas, dan US Dollar Index (DXY). 
                Prediksi mengimplementasikan algoritma Gradient Boosting Machine dan Extreme Gradient Boosting untuk Regresi.</p>
            </div>

            <section class="about-section" aria-labelledby="about-developer">
                <h2 id="about-developer" class="about-section-title">Pengembang</h2>
                <p class="about-developer-name">Aaron Sebastian</p>
                <p class="about-student-id">535230034</p>
                <p>Mahasiswa Teknik Informatika<br>
                Fakultas Teknologi Informasi Universitas Tarumanagara</p>
            </section>

            <section class="about-section" aria-labelledby="about-supervisors">
                <h2 id="about-supervisors" class="about-section-title">Pembimbing</h2>
                <div class="about-supervisors">
                    <article class="about-advisor">
                        <p class="about-advisor-name">TRI SUTRISNO S.Si., M.Sc.
                        <span class="about-advisor-id">(10816004)</span></p>
                        <p class="about-advisor-role">Pembimbing Utama</p>
                    </article>
                    <article class="about-advisor">
                        <p class="about-advisor-name">IRVAN LEWENUSA S.Kom., M.Kom.
                        <span class="about-advisor-id">(10823004)</span></p>
                        <p class="about-advisor-role">Pembimbing Pendamping</p>
                    </article>
                </div>
            </section>
        </section>
    """)

