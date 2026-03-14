
import io
import re
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

from rapidfuzz import fuzz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# PAGE
# ---------------------------------------------------------
st.set_page_config(page_title="Literature Analysis Dashboard", layout="wide")
st.title("Literature Analysis Dashboard")

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def pick_col(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return df[lower_map[name.lower()]]
    return pd.Series([""] * len(df), index=df.index)


def normalize_scopus(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "title": pick_col(df, ["Title", "Document Title"]),
        "abstract": pick_col(df, ["Abstract"]),
        "keywords": pick_col(df, ["Author Keywords", "Index Keywords", "Keywords"]),
        "year": pick_col(df, ["Year", "Publication Year"]),
        "doi": pick_col(df, ["DOI"]),
        "source": pick_col(df, ["Source title", "Source Title", "Journal"]),
        "authors": pick_col(df, ["Authors"]),
    })


def normalize_wos(df: pd.DataFrame) -> pd.DataFrame:
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


def normalize_pubmed(df: pd.DataFrame) -> pd.DataFrame:
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


def parse_pubmed_abstract_text(pubmed_csv_file, pubmed_abs_file) -> pd.DataFrame:
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
        elif title and title in ch:
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


def deduplicate_master(master: pd.DataFrame, fuzzy_threshold: int = 96) -> pd.DataFrame:
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


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)

    stop_words = {
        "the", "and", "for", "with", "that", "this", "from", "into", "their",
        "were", "have", "has", "had", "been", "being", "than", "then", "they",
        "them", "some", "such", "more", "also", "using", "used", "use", "among",
        "across", "after", "before", "during", "between", "study", "studies",
        "results", "conclusion", "method", "methods", "patients", "patient",
        "pharmacist", "pharmacists", "medication", "adherence", "counselling",
        "counseling"
    }

    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 3]
    return " ".join(words)


def extract_effect_info(text: str) -> pd.Series:
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


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    bio.seek(0)
    return bio.getvalue()


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def fixed_effect_pool(df_: pd.DataFrame):
    w = 1 / (df_["se"] ** 2)
    pooled_log = np.sum(w * df_["log_effect"]) / np.sum(w)
    pooled_se = np.sqrt(1 / np.sum(w))
    return pooled_log, pooled_se, w


def heterogeneity_stats(df_: pd.DataFrame):
    pooled_log_fe, _, w = fixed_effect_pool(df_)
    Q = np.sum(w * (df_["log_effect"] - pooled_log_fe) ** 2)
    df_q = len(df_) - 1
    I2 = max(0, ((Q - df_q) / Q) * 100) if Q > 0 and df_q > 0 else 0
    C = np.sum(w) - (np.sum(w ** 2) / np.sum(w))
    tau2 = max(0, (Q - df_q) / C) if C > 0 else 0
    return Q, df_q, I2, tau2


def random_effect_pool(df_: pd.DataFrame):
    Q, df_q, I2, tau2 = heterogeneity_stats(df_)
    w_re = 1 / (df_["se"] ** 2 + tau2)
    pooled_log = np.sum(w_re * df_["log_effect"]) / np.sum(w_re)
    pooled_se = np.sqrt(1 / np.sum(w_re))
    return pooled_log, pooled_se, w_re, Q, df_q, I2, tau2


def summarize_pool(pooled_log: float, pooled_se: float):
    effect = np.exp(pooled_log)
    low = np.exp(pooled_log - 1.96 * pooled_se)
    high = np.exp(pooled_log + 1.96 * pooled_se)
    return effect, low, high


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("Upload files")
pubmed_csv_file = st.sidebar.file_uploader("PubMed metadata CSV", type=["csv"])
pubmed_abs_file = st.sidebar.file_uploader("PubMed abstract TXT", type=["txt", "csv"])
scopus_file = st.sidebar.file_uploader("Scopus CSV", type=["csv"])
wos_file = st.sidebar.file_uploader("Web of Science TXT/CSV", type=["txt", "csv"])

fuzzy_threshold = st.sidebar.slider("Fuzzy title threshold", 85, 100, 96)
n_topics = st.sidebar.slider("Number of LDA topics", 2, 10, 5)
network_threshold = st.sidebar.slider("Network co-occurrence threshold", 2, 30, 15)

run_btn = st.sidebar.button("Run analysis")

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if run_btn:
    if pubmed_csv_file is None or pubmed_abs_file is None or scopus_file is None or wos_file is None:
        st.error("Please upload all four files before running the analysis.")
        st.stop()

    sheets = {}
    figures = {}

    try:
        pm_full = parse_pubmed_abstract_text(pubmed_csv_file, pubmed_abs_file)
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

        master_before = pd.concat([scopus_n, wos_n, pubmed_n], ignore_index=True)
        master_after = deduplicate_master(master_before, fuzzy_threshold=fuzzy_threshold)
        master_after["year"] = pd.to_numeric(master_after["year"], errors="coerce")
        master_after["clean_text"] = master_after["text"].apply(clean_text)

        sheets["pubmed_full"] = pm_full
        sheets["master_before_dedup"] = master_before
        sheets["master_after_dedup"] = master_after

    except Exception as e:
        st.exception(e)
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Data Merge",
        "Bibliometrics",
        "Text Mining",
        "Meta-analysis"
    ])

    with tab1:
        st.subheader("Merged datasets")

        c1, c2, c3 = st.columns(3)
        c1.metric("Before deduplication", len(master_before))
        c2.metric("After deduplication", len(master_after))
        c3.metric("Duplicates removed", len(master_before) - len(master_after))

        db_counts = master_after["db"].value_counts().reset_index()
        db_counts.columns = ["database", "n_records"]
        sheets["database_counts"] = db_counts

        st.markdown("**Records by database**")
        st.dataframe(db_counts, use_container_width=True)

        st.markdown("**PubMed merged preview**")
        st.dataframe(pm_full.head(10), use_container_width=True)

        st.markdown("**Final master dataset preview**")
        st.dataframe(master_after.head(20), use_container_width=True)

    with tab2:
        st.subheader("Bibliometric analysis")

        year_counts = master_after["year"].dropna().astype(int).value_counts().sort_index()
        if len(year_counts) > 0:
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            ax1.plot(year_counts.index, year_counts.values, marker="o")
            ax1.set_xlabel("Year")
            ax1.set_ylabel("Number of publications")
            ax1.set_title("Publication trend")
            plt.tight_layout()
            st.pyplot(fig1)
            figures["publication_trend.png"] = fig1

            sheets["publication_trend"] = pd.DataFrame({
                "year": year_counts.index,
                "n_publications": year_counts.values
            })

        source_counts = master_after["source"].fillna("Unknown").value_counts().head(10)
        if len(source_counts) > 0:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.barh(source_counts.index[::-1], source_counts.values[::-1])
            ax2.set_xlabel("Number of papers")
            ax2.set_ylabel("Journal")
            ax2.set_title("Top journals")
            plt.tight_layout()
            st.pyplot(fig2)
            figures["top_journals.png"] = fig2

            sheets["top_journals"] = pd.DataFrame({
                "journal": source_counts.index,
                "n_papers": source_counts.values
            })

    with tab3:
        st.subheader("Text mining")

        all_words = " ".join(master_after["clean_text"].fillna("")).split()
        freq = Counter(all_words)
        top_words = pd.DataFrame(freq.most_common(30), columns=["word", "frequency"])
        sheets["top_words"] = top_words

        st.markdown("**Top terms**")
        st.dataframe(top_words, use_container_width=True)

        if len(top_words) > 0:
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            ax3.barh(top_words["word"][::-1], top_words["frequency"][::-1])
            ax3.set_title("Top research terms")
            plt.tight_layout()
            st.pyplot(fig3)
            figures["top_terms.png"] = fig3

        docs = master_after["clean_text"].dropna()
        docs = docs[docs.str.strip() != ""]

        if len(docs) >= 5:
            vectorizer = CountVectorizer(max_df=0.9, min_df=3)
            X = vectorizer.fit_transform(docs)

            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42
            )
            lda.fit(X)

            terms = vectorizer.get_feature_names_out()
            topic_rows = []
            for i, topic in enumerate(lda.components_):
                words = [terms[j] for j in topic.argsort()[-10:]]
                topic_rows.append({
                    "topic": i + 1,
                    "top_words": ", ".join(words)
                })

            lda_topics = pd.DataFrame(topic_rows)
            sheets["lda_topics"] = lda_topics

            st.markdown("**LDA topics**")
            st.dataframe(lda_topics, use_container_width=True)

        vectorizer_net = CountVectorizer(max_features=100)
        X_net = vectorizer_net.fit_transform(master_after["clean_text"].fillna(""))
        terms_net = vectorizer_net.get_feature_names_out()
        co_matrix = (X_net.T * X_net).toarray()

        G = nx.Graph()
        for i in range(len(terms_net)):
            for j in range(i + 1, len(terms_net)):
                if co_matrix[i, j] > network_threshold:
                    G.add_edge(terms_net[i], terms_net[j], weight=co_matrix[i, j])

        if G.number_of_nodes() > 0:
            fig4, ax4 = plt.subplots(figsize=(10, 10))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, with_labels=True, node_size=90, font_size=8, ax=ax4)
            ax4.set_title("Keyword co-occurrence network")
            plt.tight_layout()
            st.pyplot(fig4)
            figures["cooccurrence_network.png"] = fig4

    with tab4:
        st.subheader("Advanced meta-analysis")

        meta_source = master_after.copy()
        meta_source[["effect_size", "lower_ci", "upper_ci"]] = meta_source["abstract"].apply(extract_effect_info)

        for col in ["effect_size", "lower_ci", "upper_ci"]:
            meta_source[col] = pd.to_numeric(meta_source[col], errors="coerce")

        meta_df = meta_source.dropna(subset=["effect_size", "lower_ci", "upper_ci"]).copy()
        meta_df = meta_df[
            (meta_df["effect_size"] > 0) &
            (meta_df["lower_ci"] > 0) &
            (meta_df["upper_ci"] > 0) &
            (meta_df["lower_ci"] <= meta_df["effect_size"]) &
            (meta_df["effect_size"] <= meta_df["upper_ci"])
        ].copy()

        st.metric("Studies with extractable effect sizes", len(meta_df))

        if len(meta_df) == 0:
            st.warning("No valid effect sizes were automatically extracted from the abstracts.")
        else:
            meta_df["year_num"] = pd.to_numeric(meta_df["year"], errors="coerce")
            meta_df["log_effect"] = np.log(meta_df["effect_size"])
            meta_df["log_lower"] = np.log(meta_df["lower_ci"])
            meta_df["log_upper"] = np.log(meta_df["upper_ci"])
            meta_df["se"] = (meta_df["log_upper"] - meta_df["log_lower"]) / 3.92
            meta_df = meta_df.replace([np.inf, -np.inf], np.nan)
            meta_df = meta_df.dropna(subset=["log_effect", "se"]).copy()
            meta_df = meta_df[meta_df["se"] > 0].copy()

            if len(meta_df) == 0:
                st.warning("No valid studies remained after log transformation.")
            else:
                pooled_log_fe, pooled_se_fe, w_fe = fixed_effect_pool(meta_df)
                pooled_fe, pooled_fe_low, pooled_fe_high = summarize_pool(pooled_log_fe, pooled_se_fe)

                pooled_log_re, pooled_se_re, w_re, Q, df_q, I2, tau2 = random_effect_pool(meta_df)
                pooled_re, pooled_re_low, pooled_re_high = summarize_pool(pooled_log_re, pooled_se_re)

                meta_df["weight_fixed"] = w_fe
                meta_df["weight_random"] = w_re
                meta_df["weight_fixed_pct"] = 100 * meta_df["weight_fixed"] / meta_df["weight_fixed"].sum()
                meta_df["weight_random_pct"] = 100 * meta_df["weight_random"] / meta_df["weight_random"].sum()

                summary_df = pd.DataFrame([{
                    "n_studies": len(meta_df),
                    "fixed_effect": pooled_fe,
                    "fixed_low": pooled_fe_low,
                    "fixed_high": pooled_fe_high,
                    "random_effect": pooled_re,
                    "random_low": pooled_re_low,
                    "random_high": pooled_re_high,
                    "Q": Q,
                    "df": df_q,
                    "I2_percent": I2,
                    "tau2": tau2
                }])

                sheets["meta_extracted_studies"] = meta_df
                sheets["meta_summary"] = summary_df

                st.dataframe(summary_df, use_container_width=True)

                forest_df = meta_df.sort_values("effect_size").reset_index(drop=True)
                y_pos = np.arange(len(forest_df))
                left_err = forest_df["effect_size"] - forest_df["lower_ci"]
                right_err = forest_df["upper_ci"] - forest_df["effect_size"]

                fig5, ax5 = plt.subplots(figsize=(9, max(6, len(forest_df) * 0.45)))
                ax5.errorbar(
                    forest_df["effect_size"],
                    y_pos,
                    xerr=[left_err.values, right_err.values],
                    fmt="o"
                )
                ax5.axvline(1, linestyle="--")
                ax5.axvline(pooled_re, linestyle=":")
                ax5.set_xscale("log")
                ax5.set_yticks(y_pos)
                ax5.set_yticklabels(forest_df["title"].astype(str).str[:60])
                ax5.set_xlabel("Effect size (log scale)")
                ax5.set_ylabel("Study")
                ax5.set_title("Forest plot with random-effects pooled estimate")
                plt.tight_layout()
                st.pyplot(fig5)
                figures["forest_plot.png"] = fig5

                fig6, ax6 = plt.subplots(figsize=(6, 6))
                ax6.scatter(meta_df["log_effect"], meta_df["se"])
                ax6.axvline(pooled_log_re, linestyle="--")
                ax6.invert_yaxis()
                ax6.set_xlabel("Log effect size")
                ax6.set_ylabel("Standard error")
                ax6.set_title("Funnel plot")
                plt.tight_layout()
                st.pyplot(fig6)
                figures["funnel_plot.png"] = fig6

                if len(meta_df) >= 3:
                    egger_df = meta_df.copy()
                    egger_df["precision"] = 1 / egger_df["se"]
                    egger_df["std_effect"] = egger_df["log_effect"] / egger_df["se"]

                    X_egger = egger_df[["precision"]].values
                    y_egger = egger_df["std_effect"].values

                    egger_model = LinearRegression()
                    egger_model.fit(X_egger, y_egger)

                    egger_summary = pd.DataFrame([{
                        "egger_intercept": egger_model.intercept_,
                        "egger_slope": egger_model.coef_[0],
                        "n_studies": len(egger_df)
                    }])
                    sheets["egger_summary"] = egger_summary
                    st.dataframe(egger_summary, use_container_width=True)

    st.divider()
    st.subheader("Download results")

    excel_bytes = to_excel_bytes(sheets)
    st.download_button(
        "Download Excel workbook",
        data=excel_bytes,
        file_name="literature_analysis_outputs.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("literature_analysis_outputs.xlsx", excel_bytes)

        for name, df_out in sheets.items():
            zf.writestr(f"{name}.csv", df_out.to_csv(index=False))

        for name, fig in figures.items():
            zf.writestr(name, fig_to_png_bytes(fig))

    st.download_button(
        "Download ZIP bundle",
        data=zip_buffer.getvalue(),
        file_name="literature_analysis_bundle.zip",
        mime="application/zip"
    )

else:
    st.info("Upload the four files in the sidebar and click 'Run analysis'.")
