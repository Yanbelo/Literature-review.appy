
import io
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

from rapidfuzz import fuzz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Literature Analysis App", layout="wide")
st.title("Literature Analysis and Meta-analysis App")

# -----------------------------
# HELPERS
# -----------------------------
def pick_col(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return df[lower_map[name.lower()]]
    return pd.Series([""] * len(df))

def normalize_scopus(df):
    return pd.DataFrame({
        "title": pick_col(df, ["Title", "Document Title"]),
        "abstract": pick_col(df, ["Abstract"]),
        "keywords": pick_col(df, ["Author Keywords", "Index Keywords", "Keywords"]),
        "year": pick_col(df, ["Year", "Publication Year"]),
        "doi": pick_col(df, ["DOI"]),
        "source": pick_col(df, ["Source title", "Source Title", "Journal"]),
        "authors": pick_col(df, ["Authors"]),
    })

def normalize_wos(df):
    kw = pick_col(df, ["DE"]).astype(str)
    kw_plus = pick_col(df, ["ID"]).astype(str)
    keywords = (kw.fillna("") + "; " + kw_plus.fillna("")).str.strip("; ").str.strip()

    return pd.DataFrame({
        "title": pick_col(df, ["TI", "Title"]),
        "abstract": pick_col(df, ["AB", "Abstract"]),
        "keywords": keywords,
        "year": pick_col(df, ["PY", "Year"]),
        "doi": pick_col(df, ["DI", "DOI"]),
        "source": pick_col(df, ["SO", "Source", "Journal"]),
        "authors": pick_col(df, ["AU", "Authors"]),
    })

def normalize_pubmed(df):
    return pd.DataFrame({
        "pmid": pick_col(df, ["PMID", "pmid"]),
        "title": pick_col(df, ["Title", "Article Title", "title", "title_from_txt"]),
        "abstract": pick_col(df, ["Abstract", "abstract"]),
        "keywords": pick_col(df, ["Keywords", "MeSH Terms", "mesh_terms"]),
        "year": pick_col(df, ["Year", "Publication Year", "PubDate"]),
        "doi": pick_col(df, ["DOI", "doi"]),
        "source": pick_col(df, ["Journal", "Source", "Source title"]),
        "authors": pick_col(df, ["Authors", "Author"]),
    })

def parse_pubmed_abstract_text(pubmed_csv_file, pubmed_abs_file):
    pm_csv = pd.read_csv(pubmed_csv_file)

    pmid_col = None
    for c in pm_csv.columns:
        if c.strip().lower() == "pmid":
            pmid_col = c
            break
    if pmid_col is None:
        raise ValueError(f"No PMID column found. Columns: {pm_csv.columns.tolist()}")

    pm_csv["PMID"] = pm_csv[pmid_col].astype(str).str.strip()

    raw = pubmed_abs_file.read().decode("utf-8", errors="replace")
    chunks = re.split(r"\n(?=\d+\.\s)", raw.strip(), flags=re.M)

    rows = []
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue

        m_pmid = re.search(r"\bPMID:\s*(\d+)\b", ch)
        if not m_pmid:
            continue
        pmid = m_pmid.group(1)

        m_doi = re.search(r"\bDOI:\s*([^\s]+)\b", ch)
        doi = m_doi.group(1).strip() if m_doi else ""

        lines = [ln.strip() for ln in ch.splitlines()]
        if lines and re.match(r"^\d+\.\s", lines[0]):
            lines[0] = re.sub(r"^\d+\.\s*", "", lines[0]).strip()

        title = ""
        for ln in lines[1:15]:
            if not ln:
                continue
            if ln.startswith((
                "Author information:", "BACKGROUND:", "RESULTS:", "CONCLUSIONS:",
                "Copyright", "DOI:", "PMID:", "PMCID:", "Conflict of interest statement:"
            )):
                continue
            title = ln
            break

        abstract = ""
        if "Author information:" in ch:
            after_author = ch.split("Author information:", 1)[1]
            abstract = re.split(
                r"\n(?:DOI:|PMID:|PMCID:|Copyright|Conflict of interest statement:)",
                after_author,
                maxsplit=1
            )[0].strip()
        else:
            if title and title in ch:
                after_title = ch.split(title, 1)[1]
                abstract = re.split(
                    r"\n(?:DOI:|PMID:|PMCID:|Copyright|Conflict of interest statement:)",
                    after_title,
                    maxsplit=1
                )[0].strip()

        abstract = re.sub(r"\s+", " ", abstract).strip()

        rows.append({
            "PMID": pmid,
            "doi_from_txt": doi,
            "title_from_txt": title,
            "abstract": abstract
        })

    pm_abs = pd.DataFrame(rows).drop_duplicates(subset=["PMID"])

    pm_full = pm_csv.merge(
        pm_abs[["PMID", "abstract", "doi_from_txt", "title_from_txt"]],
        on="PMID",
        how="left"
    )

    doi_col = None
    for c in pm_full.columns:
        if c.strip().lower() == "doi":
            doi_col = c
            break

    if doi_col:
        pm_full[doi_col] = pm_full[doi_col].astype(str).replace({"nan": ""})
        pm_full[doi_col] = pm_full[doi_col].where(
            pm_full[doi_col].str.strip() != "",
            pm_full["doi_from_txt"]
        )

    return pm_full

def deduplicate_master(master, fuzzy_threshold=96):
    master = master.copy()

    master["doi"] = master["doi"].astype(str).str.lower().str.strip()
    master["doi"] = master["doi"].replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})

    with_doi = master[~master["doi"].isna()].drop_duplicates(subset=["doi"], keep="first")
    no_doi = master[master["doi"].isna()].copy()

    combined = pd.concat([with_doi, no_doi], ignore_index=False)

    combined["title_clean"] = (
        combined["title"].astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.strip()
    )

    no_doi = combined[combined["doi"].isna()].copy().reset_index(drop=False)
    to_drop = set()
    titles = no_doi["title_clean"].tolist()

    for i in range(len(titles)):
        if i in to_drop:
            continue
        for j in range(i + 1, len(titles)):
            if j in to_drop:
                continue
            if fuzz.ratio(titles[i], titles[j]) >= fuzzy_threshold:
                to_drop.add(j)

    drop_idx = no_doi.loc[list(to_drop), "index"].tolist()
    combined = combined.drop(index=drop_idx)

    combined["text"] = (
        combined["title"].fillna("").astype(str) + " " +
        combined["abstract"].fillna("").astype(str) + " " +
        combined["keywords"].fillna("").astype(str)
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    return combined

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)

    basic_stop = {
        "the","and","for","with","that","this","from","into","their","were","have",
        "has","had","been","being","than","then","they","them","some","such","more",
        "also","using","used","use","among","across","after","before","during","between",
        "study","studies","results","conclusion","method","methods",
        "patients","patient","pharmacist","pharmacists",
        "medication","adherence","counselling","counseling"
    }

    words = text.split()
    words = [w for w in words if w not in basic_stop and len(w) > 3]
    return " ".join(words)

def extract_effect_info(text):
    text = str(text)
    patterns = [
        r"(?:odds ratio|adjusted odds ratio|aOR|OR)\s*[:=]?\s*\(?\s*([0-9]*\.?[0-9]+)\)?[^\.;\n]*?(?:95%\s*CI|CI)\s*[:=]?\s*\(?\s*([0-9]*\.?[0-9]+)\s*(?:-|–|to|,)\s*([0-9]*\.?[0-9]+)\)?",
        r"(?:risk ratio|relative risk|RR)\s*[:=]?\s*\(?\s*([0-9]*\.?[0-9]+)\)?[^\.;\n]*?(?:95%\s*CI|CI)\s*[:=]?\s*\(?\s*([0-9]*\.?[0-9]+)\s*(?:-|–|to|,)\s*([0-9]*\.?[0-9]+)\)?",
        r"(?:hazard ratio|HR)\s*[:=]?\s*\(?\s*([0-9]*\.?[0-9]+)\)?[^\.;\n]*?(?:95%\s*CI|CI)\s*[:=]?\s*\(?\s*([0-9]*\.?[0-9]+)\s*(?:-|–|to|,)\s*([0-9]*\.?[0-9]+)\)?",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            return pd.Series([m.group(1), m.group(2), m.group(3)])
    return pd.Series([np.nan, np.nan, np.nan])

def dataframe_to_excel_bytes(sheets):
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    bio.seek(0)
    return bio.getvalue()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Upload files")
pubmed_csv_file = st.sidebar.file_uploader("PubMed metadata CSV", type=["csv"])
pubmed_abs_file = st.sidebar.file_uploader("PubMed abstract TXT", type=["txt", "csv"])
scopus_file = st.sidebar.file_uploader("Scopus CSV", type=["csv"])
wos_file = st.sidebar.file_uploader("Web of Science TXT/CSV", type=["txt", "csv"])
run_btn = st.sidebar.button("Run analysis")

# -----------------------------
# MAIN
# -----------------------------
if run_btn:
    if pubmed_csv_file is None or pubmed_abs_file is None or scopus_file is None or wos_file is None:
        st.error("Please upload PubMed CSV, PubMed abstract TXT, Scopus CSV, and WoS TXT/CSV.")
        st.stop()

    sheets = {}
    images = {}

    # 1. PubMed merge
    pm_full = parse_pubmed_abstract_text(pubmed_csv_file, pubmed_abs_file)
    sheets["pubmed_full"] = pm_full
    st.subheader("Merged PubMed records")
    st.write(pm_full.head())

    # 2. Load Scopus/WoS
    scopus = pd.read_csv(scopus_file)
    wos = pd.read_csv(
        wos_file,
        sep=None,
        engine="python",
        quoting=3,
        on_bad_lines="skip"
    )

    scopus_n = normalize_scopus(scopus)
    scopus_n["db"] = "scopus"

    wos_n = normalize_wos(wos)
    wos_n["db"] = "wos"

    pubmed_n = normalize_pubmed(pm_full)
    pubmed_n["db"] = "pubmed"

    master = pd.concat([scopus_n, wos_n, pubmed_n], ignore_index=True)
    st.subheader("Master dataset")
    st.write("Before deduplication:", len(master))

    master = deduplicate_master(master, fuzzy_threshold=96)
    st.write("After deduplication:", len(master))
    st.dataframe(master.head())
    sheets["master_literature"] = master

    # 3. Publication trend
    master["year"] = pd.to_numeric(master["year"], errors="coerce")
    year_counts = master["year"].value_counts().sort_index()

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(year_counts.index, year_counts.values, marker="o")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of publications")
    ax1.set_title("Publication trend")
    st.pyplot(fig1)
    images["publication_trend.png"] = fig1

    # 4. Top journals
    source_counts = master["source"].fillna("Unknown").value_counts().head(10)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.barh(source_counts.index[::-1], source_counts.values[::-1])
    ax2.set_xlabel("Number of papers")
    ax2.set_ylabel("Journal")
    ax2.set_title("Top journals")
    st.pyplot(fig2)
    images["top_journals.png"] = fig2

    # 5. Text cleaning
    master["clean_text"] = master["text"].apply(clean_text)
    all_words = " ".join(master["clean_text"]).split()
    freq = Counter(all_words)

    top_words = pd.DataFrame(freq.most_common(30), columns=["word", "frequency"])
    sheets["top_words"] = top_words
    st.subheader("Top research terms")
    st.dataframe(top_words)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.barh(top_words["word"][::-1], top_words["frequency"][::-1])
    ax3.set_title("Top research terms")
    st.pyplot(fig3)
    images["top_terms.png"] = fig3

    # 6. Topic modelling
    docs = master["clean_text"].dropna()
    docs = docs[docs.str.strip() != ""]
    if len(docs) >= 5:
        vectorizer = CountVectorizer(max_df=0.9, min_df=3)
        X = vectorizer.fit_transform(docs)

        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda.fit(X)

        terms = vectorizer.get_feature_names_out()
        topic_rows = []
        for i, topic in enumerate(lda.components_):
            words = [terms[j] for j in topic.argsort()[-10:]]
            topic_rows.append({"topic": i + 1, "top_words": ", ".join(words)})

        lda_topics = pd.DataFrame(topic_rows)
        sheets["lda_topics"] = lda_topics
        st.subheader("LDA topics")
        st.dataframe(lda_topics)

    # 7. Co-occurrence network
    vectorizer_net = CountVectorizer(max_features=100)
    X_net = vectorizer_net.fit_transform(master["clean_text"].fillna(""))
    terms_net = vectorizer_net.get_feature_names_out()
    co_matrix = (X_net.T * X_net).toarray()

    G = nx.Graph()
    for i in range(len(terms_net)):
        for j in range(i + 1, len(terms_net)):
            if co_matrix[i, j] > 15:
                G.add_edge(terms_net[i], terms_net[j], weight=co_matrix[i, j])

    if G.number_of_nodes() > 0:
        fig4, ax4 = plt.subplots(figsize=(10, 10))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=80, font_size=8, ax=ax4)
        ax4.set_title("Keyword co-occurrence network")
        st.pyplot(fig4)
        images["cooccurrence_network.png"] = fig4

    # 8. Meta-analysis extraction
    master[["effect_size", "lower_ci", "upper_ci"]] = master["abstract"].apply(extract_effect_info)
    for col in ["effect_size", "lower_ci", "upper_ci"]:
        master[col] = pd.to_numeric(master[col], errors="coerce")

    meta_df = master.dropna(subset=["effect_size", "lower_ci", "upper_ci"]).copy()
    meta_df = meta_df[
        (meta_df["effect_size"] > 0) &
        (meta_df["lower_ci"] > 0) &
        (meta_df["upper_ci"] > 0) &
        (meta_df["lower_ci"] <= meta_df["effect_size"]) &
        (meta_df["effect_size"] <= meta_df["upper_ci"])
    ].copy()

    if len(meta_df) > 0:
        meta_df["log_effect"] = np.log(meta_df["effect_size"])
        meta_df["log_lower"] = np.log(meta_df["lower_ci"])
        meta_df["log_upper"] = np.log(meta_df["upper_ci"])
        meta_df["se"] = (meta_df["log_upper"] - meta_df["log_lower"]) / 3.92
        meta_df = meta_df.replace([np.inf, -np.inf], np.nan)
        meta_df = meta_df.dropna(subset=["log_effect", "se"])
        meta_df = meta_df[meta_df["se"] > 0].copy()

        meta_df["weight_fixed"] = 1 / (meta_df["se"] ** 2)

        if meta_df["weight_fixed"].sum() > 0:
            pooled_log = np.sum(meta_df["weight_fixed"] * meta_df["log_effect"]) / np.sum(meta_df["weight_fixed"])
            pooled_se = np.sqrt(1 / np.sum(meta_df["weight_fixed"]))

            pooled_effect = np.exp(pooled_log)
            pooled_low = np.exp(pooled_log - 1.96 * pooled_se)
            pooled_high = np.exp(pooled_log + 1.96 * pooled_se)

            Q = np.sum(meta_df["weight_fixed"] * (meta_df["log_effect"] - pooled_log) ** 2)
            df_q = len(meta_df) - 1
            I2 = max(0, ((Q - df_q) / Q) * 100) if Q > 0 and df_q > 0 else 0

            C = np.sum(meta_df["weight_fixed"]) - (
                np.sum(meta_df["weight_fixed"] ** 2) / np.sum(meta_df["weight_fixed"])
            )
            tau2 = max(0, (Q - df_q) / C) if C > 0 else 0

            meta_df["weight_random"] = 1 / (meta_df["se"] ** 2 + tau2)
            pooled_log_random = np.sum(meta_df["weight_random"] * meta_df["log_effect"]) / np.sum(meta_df["weight_random"])
            pooled_se_random = np.sqrt(1 / np.sum(meta_df["weight_random"]))

            pooled_random = np.exp(pooled_log_random)
            pooled_random_low = np.exp(pooled_log_random - 1.96 * pooled_se_random)
            pooled_random_high = np.exp(pooled_log_random + 1.96 * pooled_se_random)

            summary_df = pd.DataFrame([{
                "n_studies": len(meta_df),
                "fixed_effect": pooled_effect,
                "fixed_low": pooled_low,
                "fixed_high": pooled_high,
                "Q": Q,
                "I2_percent": I2,
                "tau2": tau2,
                "random_effect": pooled_random,
                "random_low": pooled_random_low,
                "random_high": pooled_random_high
            }])

            sheets["meta_extracted_studies"] = meta_df
            sheets["meta_summary"] = summary_df

            st.subheader("Meta-analysis summary")
            st.dataframe(summary_df)

            # Forest plot
            forest_df = meta_df.sort_values("effect_size").reset_index(drop=True)
            y_pos = np.arange(len(forest_df))
            left_err = forest_df["effect_size"] - forest_df["lower_ci"]
            right_err = forest_df["upper_ci"] - forest_df["effect_size"]

            fig5, ax5 = plt.subplots(figsize=(8, max(6, len(forest_df) * 0.45)))
            ax5.errorbar(
                forest_df["effect_size"],
                y_pos,
                xerr=[left_err.values, right_err.values],
                fmt="o"
            )
            ax5.axvline(1, linestyle="--")
            ax5.set_xscale("log")
            ax5.set_yticks(y_pos)
            ax5.set_yticklabels(forest_df["title"].astype(str).str[:60])
            ax5.set_xlabel("Effect size (log scale)")
            ax5.set_ylabel("Study")
            ax5.set_title("Forest plot of extracted studies")
            plt.tight_layout()
            st.pyplot(fig5)
            images["forest_plot.png"] = fig5

            # Funnel plot
            fig6, ax6 = plt.subplots(figsize=(6, 6))
            ax6.scatter(meta_df["log_effect"], meta_df["se"])
            ax6.invert_yaxis()
            ax6.set_xlabel("Log effect size")
            ax6.set_ylabel("Standard error")
            ax6.set_title("Funnel plot")
            plt.tight_layout()
            st.pyplot(fig6)
            images["funnel_plot.png"] = fig6

    # 9. Downloads
    excel_bytes = dataframe_to_excel_bytes(sheets)

    st.download_button(
        "Download Excel results",
        data=excel_bytes,
        file_name="literature_analysis_outputs.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("literature_analysis_outputs.xlsx", excel_bytes)

        for name, df_out in sheets.items():
            zf.writestr(f"{name}.csv", df_out.to_csv(index=False))

        for fname, fig in images.items():
            img = io.BytesIO()
            fig.savefig(img, format="png", dpi=200, bbox_inches="tight")
            img.seek(0)
            zf.writestr(fname, img.read())

    st.download_button(
        "Download ZIP bundle",
        data=zip_buffer.getvalue(),
        file_name="literature_analysis_bundle.zip",
        mime="application/zip"
    )

else:
    st.info("Upload the four files in the sidebar, then click Run analysis.")
