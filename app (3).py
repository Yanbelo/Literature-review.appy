
import io
import re
import zipfile
from collections import Counter
from math import erf, sqrt

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

from rapidfuzz import fuzz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Literature Analysis Dashboard", layout="wide")
st.title("Literature Analysis Dashboard")

# =========================================================
# HELPERS
# =========================================================
def pick_col(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return df[lower_map[name.lower()]]
    return pd.Series([""] * len(df), index=df.index)


def detect_optional_column(df: pd.DataFrame, candidates: list[str]):
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def normalize_scopus(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "title": pick_col(df, ["Title", "Document Title"]),
        "abstract": pick_col(df, ["Abstract"]),
        "keywords": pick_col(df, ["Author Keywords", "Index Keywords", "Keywords"]),
        "year": pick_col(df, ["Year", "Publication Year"]),
        "doi": pick_col(df, ["DOI"]),
        "source": pick_col(df, ["Source title", "Source Title", "Journal"]),
        "authors": pick_col(df, ["Authors"]),
    })
    cited_col = detect_optional_column(df, ["Cited by", "Times Cited", "TC", "Citations"])
    out["citations"] = pd.to_numeric(df[cited_col], errors="coerce") if cited_col else np.nan
    country_col = detect_optional_column(df, ["Country", "Countries", "Affiliations", "Addresses", "C1"])
    out["affiliations"] = df[country_col] if country_col else ""
    return out


def normalize_wos(df: pd.DataFrame) -> pd.DataFrame:
    kw = pick_col(df, ["DE"]).astype(str)
    kw_plus = pick_col(df, ["ID"]).astype(str)
    keywords = (kw.fillna("") + "; " + kw_plus.fillna("")).str.strip("; ").str.strip()

    out = pd.DataFrame({
        "title": pick_col(df, ["TI", "Title"]),
        "abstract": pick_col(df, ["AB", "Abstract"]),
        "keywords": keywords,
        "year": pick_col(df, ["PY", "Year"]),
        "doi": pick_col(df, ["DI", "DOI"]),
        "source": pick_col(df, ["SO", "Source", "Journal"]),
        "authors": pick_col(df, ["AU", "Authors"]),
    })
    cited_col = detect_optional_column(df, ["Times Cited", "TC", "Citations"])
    out["citations"] = pd.to_numeric(df[cited_col], errors="coerce") if cited_col else np.nan
    country_col = detect_optional_column(df, ["C1", "Addresses", "Affiliations"])
    out["affiliations"] = df[country_col] if country_col else ""
    return out


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
        "citations": np.nan,
        "affiliations": pick_col(df, ["Affiliations", "Address", "Addresses"]),
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

    return combined.reset_index(drop=True)


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


def filter_master_by_keywords(df: pd.DataFrame, keyword_query: str, fields: list[str]) -> pd.DataFrame:
    if not keyword_query or not keyword_query.strip():
        return df.copy()

    terms = [t.strip() for t in keyword_query.split(",") if t.strip()]
    if not terms:
        return df.copy()

    search_text = pd.Series("", index=df.index, dtype="object")
    for field in fields:
        if field in df.columns:
            search_text = search_text + " " + df[field].fillna("").astype(str)
    search_text = search_text.str.lower()

    mask = pd.Series(False, index=df.index)
    for term in terms:
        mask = mask | search_text.str.contains(re.escape(term.lower()), na=False, regex=True)

    return df.loc[mask].copy()


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def p_value_from_z(z: float) -> float:
    return 2 * (1 - normal_cdf(abs(z)))


def fixed_effect_pool(df_: pd.DataFrame):
    w = 1 / (df_["se"] ** 2)
    pooled_log = np.sum(w * df_["log_effect"]) / np.sum(w)
    pooled_se = np.sqrt(1 / np.sum(w))
    z = pooled_log / pooled_se if pooled_se > 0 else np.nan
    p = p_value_from_z(z) if pd.notna(z) else np.nan
    return pooled_log, pooled_se, w, z, p


def heterogeneity_stats(df_: pd.DataFrame):
    pooled_log_fe, _, w, _, _ = fixed_effect_pool(df_)
    Q = np.sum(w * (df_["log_effect"] - pooled_log_fe) ** 2)
    df_q = len(df_) - 1
    I2 = max(0, ((Q - df_q) / Q) * 100) if Q > 0 and df_q > 0 else 0
    H2 = Q / df_q if df_q > 0 else np.nan
    C = np.sum(w) - (np.sum(w ** 2) / np.sum(w))
    tau2 = max(0, (Q - df_q) / C) if C > 0 else 0
    return Q, df_q, I2, H2, tau2


def random_effect_pool(df_: pd.DataFrame):
    Q, df_q, I2, H2, tau2 = heterogeneity_stats(df_)
    w_re = 1 / (df_["se"] ** 2 + tau2)
    pooled_log = np.sum(w_re * df_["log_effect"]) / np.sum(w_re)
    pooled_se = np.sqrt(1 / np.sum(w_re))
    z = pooled_log / pooled_se if pooled_se > 0 else np.nan
    p = p_value_from_z(z) if pd.notna(z) else np.nan
    return pooled_log, pooled_se, w_re, Q, df_q, I2, H2, tau2, z, p


def summarize_pool(pooled_log: float, pooled_se: float):
    effect = np.exp(pooled_log)
    low = np.exp(pooled_log - 1.96 * pooled_se)
    high = np.exp(pooled_log + 1.96 * pooled_se)
    return effect, low, high


def prediction_interval_random(pooled_log: float, tau2: float):
    pred_low = np.exp(pooled_log - 1.96 * np.sqrt(tau2))
    pred_high = np.exp(pooled_log + 1.96 * np.sqrt(tau2))
    return pred_low, pred_high


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


def bradford_zones(df: pd.DataFrame) -> pd.DataFrame:
    src = df["source"].fillna("Unknown").astype(str)
    counts = src.value_counts().reset_index()
    counts.columns = ["journal", "n_papers"]
    total = counts["n_papers"].sum()
    target = total / 3 if total > 0 else 0

    zones = []
    cum = 0
    current_zone = 1
    for _, row in counts.iterrows():
        if cum >= current_zone * target and current_zone < 3:
            current_zone += 1
        zones.append(current_zone)
        cum += row["n_papers"]

    counts["bradford_zone"] = zones
    counts["cum_papers"] = counts["n_papers"].cumsum()
    return counts


def build_prisma_counts(before_df: pd.DataFrame, after_dedup_df: pd.DataFrame, filtered_df: pd.DataFrame) -> pd.DataFrame:
    before_total = len(before_df)
    after_dedup = len(after_dedup_df)
    duplicates_removed = before_total - after_dedup
    after_filter = len(filtered_df)
    filter_excluded = after_dedup - after_filter

    return pd.DataFrame({
        "stage": [
            "Records identified",
            "Duplicates removed",
            "Records after deduplication",
            "Records excluded by filter",
            "Records retained for analysis"
        ],
        "count": [
            before_total,
            duplicates_removed,
            after_dedup,
            filter_excluded,
            after_filter
        ]
    })


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Upload files")
pubmed_csv_file = st.sidebar.file_uploader("PubMed metadata CSV", type=["csv"])
pubmed_abs_file = st.sidebar.file_uploader("PubMed abstract TXT", type=["txt", "csv"])
scopus_file = st.sidebar.file_uploader("Scopus CSV", type=["csv"])
wos_file = st.sidebar.file_uploader("Web of Science TXT/CSV", type=["txt", "csv"])

st.sidebar.subheader("Paper filter")
keyword_query = st.sidebar.text_input(
    "Filter papers by keyword(s)",
    placeholder="e.g. adherence, intervention, diabetes"
)
filter_fields = st.sidebar.multiselect(
    "Search in fields",
    options=["title", "abstract", "keywords", "source", "authors", "text"],
    default=["title", "abstract", "keywords"]
)

fuzzy_threshold = st.sidebar.slider("Fuzzy title threshold", 85, 100, 96)
n_topics = st.sidebar.slider("Number of LDA topics", 2, 10, 5)
network_threshold = st.sidebar.slider("Keyword network threshold", 2, 30, 15)
citation_threshold = st.sidebar.slider("Citation network threshold", 1, 20, 3)

run_btn = st.sidebar.button("Run analysis")

# =========================================================
# MAIN
# =========================================================
if run_btn:
    if pubmed_csv_file is None or pubmed_abs_file is None or scopus_file is None or wos_file is None:
        st.error("Please upload all four files before running the analysis.")
        st.stop()

    sheets = {}
    figures = {}

    try:
        pm_full = parse_pubmed_abstract_text(pubmed_csv_file, pubmed_abs_file)
        scopus = pd.read_csv(scopus_file)
        wos = pd.read_csv(wos_file, sep=None, engine="python", quoting=3, on_bad_lines="skip")

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

        filtered_master = filter_master_by_keywords(
            master_after,
            keyword_query=keyword_query,
            fields=filter_fields
        )
        if len(filtered_master) > 0:
            filtered_master["clean_text"] = filtered_master["text"].apply(clean_text)

        sheets["pubmed_full"] = pm_full
        sheets["master_before_dedup"] = master_before
        sheets["master_after_dedup"] = master_after
        sheets["master_filtered"] = filtered_master

        prisma_df = build_prisma_counts(master_before, master_after, filtered_master)
        sheets["prisma_counts"] = prisma_df

    except Exception as e:
        st.exception(e)
        st.stop()

    st.write(f"Filtered papers: {len(filtered_master)} / {len(master_after)}")
    if keyword_query.strip():
        st.caption(f"Active keyword filter: {keyword_query}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Data Merge",
        "Bibliometrics",
        "Text Mining",
        "Meta-analysis"
    ])

    with tab1:
        st.subheader("Merged datasets")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Before deduplication", len(master_before))
        c2.metric("After deduplication", len(master_after))
        c3.metric("After filter", len(filtered_master))
        c4.metric("Duplicates removed", len(master_before) - len(master_after))

        st.markdown("### PRISMA counts panel")
        st.dataframe(prisma_df, use_container_width=True)

        fig_prisma, ax_prisma = plt.subplots(figsize=(8, 4))
        ax_prisma.barh(prisma_df["stage"][::-1], prisma_df["count"][::-1])
        ax_prisma.set_xlabel("Count")
        ax_prisma.set_title("PRISMA-style flow counts")
        plt.tight_layout()
        st.pyplot(fig_prisma)
        figures["prisma_counts.png"] = fig_prisma

        db_counts = filtered_master["db"].value_counts().reset_index()
        db_counts.columns = ["database", "n_records"]
        sheets["database_counts_filtered"] = db_counts

        st.markdown("**Records by database**")
        st.dataframe(db_counts, use_container_width=True)

        st.markdown("**PubMed merged preview**")
        st.dataframe(pm_full.head(10), use_container_width=True)

        st.markdown("**Final filtered dataset preview**")
        st.dataframe(filtered_master.head(20), use_container_width=True)

    with tab2:
        st.subheader("Bibliometric analysis")
        bib_df = filtered_master.copy()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total papers", len(bib_df))
        c2.metric("Unique journals", bib_df["source"].fillna("").nunique())
        c3.metric("Unique DOIs", bib_df["doi"].dropna().nunique())
        c4.metric("Databases", bib_df["db"].fillna("").nunique())

        year_counts = bib_df["year"].dropna().astype(int).value_counts().sort_index()
        if len(year_counts) > 0:
            st.markdown("### Publications by year")
            fig_year, ax_year = plt.subplots(figsize=(8, 5))
            ax_year.plot(year_counts.index, year_counts.values, marker="o")
            ax_year.set_xlabel("Year")
            ax_year.set_ylabel("Number of publications")
            ax_year.set_title("Publication trend")
            plt.tight_layout()
            st.pyplot(fig_year)
            figures["publication_trend.png"] = fig_year
            sheets["publication_trend"] = pd.DataFrame({"year": year_counts.index, "n_publications": year_counts.values})

        db_counts = bib_df["db"].fillna("Unknown").value_counts()
        st.markdown("### Database contribution")
        db_df = pd.DataFrame({"database": db_counts.index, "n_records": db_counts.values})
        sheets["database_contribution"] = db_df
        st.dataframe(db_df, use_container_width=True)

        fig_db, ax_db = plt.subplots(figsize=(7, 4))
        ax_db.bar(db_counts.index, db_counts.values)
        ax_db.set_xlabel("Database")
        ax_db.set_ylabel("Number of papers")
        ax_db.set_title("Database contribution")
        plt.tight_layout()
        st.pyplot(fig_db)
        figures["database_contribution.png"] = fig_db

        source_counts = bib_df["source"].fillna("Unknown").value_counts().head(15)
        if len(source_counts) > 0:
            st.markdown("### Top journals")
            top_journals_df = pd.DataFrame({"journal": source_counts.index, "n_papers": source_counts.values})
            sheets["top_journals"] = top_journals_df
            st.dataframe(top_journals_df, use_container_width=True)

            fig_j, ax_j = plt.subplots(figsize=(9, 6))
            ax_j.barh(source_counts.index[::-1], source_counts.values[::-1])
            ax_j.set_xlabel("Number of papers")
            ax_j.set_ylabel("Journal")
            ax_j.set_title("Top journals")
            plt.tight_layout()
            st.pyplot(fig_j)
            figures["top_journals.png"] = fig_j

        author_counts = (
            bib_df["authors"]
            .dropna()
            .astype(str)
            .str.split(";|,")
            .explode()
            .str.strip()
        )
        author_counts = author_counts[author_counts != ""].value_counts().head(20)
        if len(author_counts) > 0:
            st.markdown("### Top authors")
            top_authors_df = pd.DataFrame({"author": author_counts.index, "n_records": author_counts.values})
            sheets["top_authors"] = top_authors_df
            st.dataframe(top_authors_df, use_container_width=True)

        bib_df["n_authors"] = (
            bib_df["authors"].fillna("").astype(str)
            .apply(lambda x: len([a.strip() for a in re.split(r";|,", x) if a.strip()]))
        )
        authors_per_paper = bib_df["n_authors"].value_counts().sort_index()
        st.markdown("### Authors per paper distribution")
        app_df = pd.DataFrame({"n_authors": authors_per_paper.index, "n_papers": authors_per_paper.values})
        sheets["authors_per_paper"] = app_df
        st.dataframe(app_df, use_container_width=True)

        keyword_series = (
            bib_df["keywords"].fillna("").astype(str)
            .str.split(";|,")
            .explode().str.strip()
        )
        keyword_series = keyword_series[keyword_series != ""]
        keyword_counts = keyword_series.value_counts().head(25)
        if len(keyword_counts) > 0:
            st.markdown("### Most frequent keywords")
            kw_df = pd.DataFrame({"keyword": keyword_counts.index, "frequency": keyword_counts.values})
            sheets["keyword_frequency"] = kw_df
            st.dataframe(kw_df, use_container_width=True)

        st.markdown("### Bradford-like journal zone summary")
        bradford_df = bradford_zones(bib_df)
        sheets["bradford_zones"] = bradford_df
        st.dataframe(bradford_df.head(30), use_container_width=True)

        fig_brad, ax_brad = plt.subplots(figsize=(8, 5))
        zone_counts = bradford_df.groupby("bradford_zone")["n_papers"].sum()
        ax_brad.bar(zone_counts.index.astype(str), zone_counts.values)
        ax_brad.set_xlabel("Bradford zone")
        ax_brad.set_ylabel("Papers")
        ax_brad.set_title("Bradford-like zone distribution")
        plt.tight_layout()
        st.pyplot(fig_brad)
        figures["bradford_zones.png"] = fig_brad

        trend_source = bib_df.dropna(subset=["year"]).copy()
        if len(trend_source) > 0:
            top_sources = bib_df["source"].fillna("Unknown").value_counts().head(5).index.tolist()
            trend_source["source_group"] = trend_source["source"].fillna("Unknown")
            trend_source = trend_source[trend_source["source_group"].isin(top_sources)]
            source_year = (
                trend_source.groupby(["year", "source_group"])
                .size().reset_index(name="n")
                .sort_values(["year", "source_group"])
            )
            sheets["source_year_trend"] = source_year
            st.markdown("### Source contribution by year")
            st.dataframe(source_year, use_container_width=True)

        citation_col_present = bib_df["citations"].notna().any()
        if citation_col_present:
            cited_df = bib_df.dropna(subset=["citations"]).sort_values("citations", ascending=False).head(20)
            if len(cited_df) > 0:
                st.markdown("### Top cited papers")
                top_cited = cited_df[["title", "source", "year", "citations"]].copy()
                sheets["top_cited_papers"] = top_cited
                st.dataframe(top_cited, use_container_width=True)

        if bib_df["affiliations"].fillna("").astype(str).str.len().sum() > 0:
            aff_series = (
                bib_df["affiliations"].fillna("").astype(str)
                .str.split(";|,")
                .explode().str.strip()
            )
            aff_series = aff_series[aff_series != ""].value_counts().head(20)
            if len(aff_series) > 0:
                st.markdown("### Top countries / affiliations")
                aff_df = pd.DataFrame({"country_or_affiliation": aff_series.index, "n_records": aff_series.values})
                sheets["country_affiliation_summary"] = aff_df
                st.dataframe(aff_df, use_container_width=True)

    with tab3:
        st.subheader("Text mining")
        text_df = filtered_master.copy()

        all_words = " ".join(text_df["clean_text"].fillna("")).split()
        freq = Counter(all_words)
        top_words = pd.DataFrame(freq.most_common(30), columns=["word", "frequency"])
        sheets["top_words"] = top_words

        st.markdown("### Top terms")
        st.dataframe(top_words, use_container_width=True)

        if len(top_words) > 0:
            fig_terms, ax_terms = plt.subplots(figsize=(10, 6))
            ax_terms.barh(top_words["word"][::-1], top_words["frequency"][::-1])
            ax_terms.set_title("Top research terms")
            plt.tight_layout()
            st.pyplot(fig_terms)
            figures["top_terms.png"] = fig_terms

        docs = text_df["clean_text"].dropna()
        docs = docs[docs.str.strip() != ""]

        if len(docs) >= 5:
            vectorizer = CountVectorizer(max_df=0.9, min_df=3)
            X = vectorizer.fit_transform(docs)
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(X)

            terms = vectorizer.get_feature_names_out()
            topic_rows = []
            for i, topic in enumerate(lda.components_):
                words = [terms[j] for j in topic.argsort()[-10:]]
                topic_rows.append({"topic": i + 1, "top_words": ", ".join(words)})
            lda_topics = pd.DataFrame(topic_rows)
            sheets["lda_topics"] = lda_topics

            st.markdown("### LDA topics")
            st.dataframe(lda_topics, use_container_width=True)

        vectorizer_net = CountVectorizer(max_features=100)
        X_net = vectorizer_net.fit_transform(text_df["clean_text"].fillna(""))
        terms_net = vectorizer_net.get_feature_names_out()
        co_matrix = (X_net.T * X_net).toarray()

        G = nx.Graph()
        for i in range(len(terms_net)):
            for j in range(i + 1, len(terms_net)):
                if co_matrix[i, j] > network_threshold:
                    G.add_edge(terms_net[i], terms_net[j], weight=co_matrix[i, j])

        if G.number_of_nodes() > 0:
            fig_net, ax_net = plt.subplots(figsize=(10, 10))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, with_labels=True, node_size=90, font_size=8, ax=ax_net)
            ax_net.set_title("Keyword co-occurrence network")
            plt.tight_layout()
            st.pyplot(fig_net)
            figures["keyword_network.png"] = fig_net

            edges_df = pd.DataFrame(
                [{"term1": u, "term2": v, "weight": d["weight"]} for u, v, d in G.edges(data=True)]
            ).sort_values("weight", ascending=False)
            sheets["keyword_network_edges"] = edges_df

        trend_df = text_df.dropna(subset=["year"]).copy()
        if len(trend_df) > 0:
            st.markdown("### Keyword thematic evolution")
            top_evo_terms = keyword_counts.index[:8].tolist() if 'keyword_counts' in locals() and len(keyword_counts) > 0 else []
            if not top_evo_terms:
                top_evo_terms = [w for w in top_words["word"].head(8).tolist()]

            evolution_rows = []
            search_text = (
                trend_df["title"].fillna("").astype(str) + " " +
                trend_df["abstract"].fillna("").astype(str) + " " +
                trend_df["keywords"].fillna("").astype(str)
            ).str.lower()

            for term in top_evo_terms:
                mask = search_text.str.contains(re.escape(str(term).lower()), na=False, regex=True)
                tmp = trend_df.loc[mask].groupby("year").size().reset_index(name="n")
                tmp["term"] = term
                evolution_rows.append(tmp)

            if evolution_rows:
                evolution_df = pd.concat(evolution_rows, ignore_index=True)
                sheets["keyword_thematic_evolution"] = evolution_df
                st.dataframe(evolution_df, use_container_width=True)

                fig_evo, ax_evo = plt.subplots(figsize=(9, 5))
                for term in evolution_df["term"].unique():
                    sub = evolution_df[evolution_df["term"] == term]
                    ax_evo.plot(sub["year"], sub["n"], marker="o", label=term)
                ax_evo.set_xlabel("Year")
                ax_evo.set_ylabel("Number of papers")
                ax_evo.set_title("Keyword thematic evolution")
                ax_evo.legend(fontsize=8)
                plt.tight_layout()
                st.pyplot(fig_evo)
                figures["keyword_thematic_evolution.png"] = fig_evo

        if text_df["citations"].notna().sum() > 0:
            st.markdown("### Citation network")
            cited_sub = text_df.dropna(subset=["citations"]).copy()
            cited_sub["citations"] = pd.to_numeric(cited_sub["citations"], errors="coerce")
            cited_sub = cited_sub.dropna(subset=["citations"]).sort_values("citations", ascending=False).head(40)

            Gc = nx.Graph()
            for _, row in cited_sub.iterrows():
                Gc.add_node(str(row["title"])[:60], citations=float(row["citations"]))

            for i in range(len(cited_sub)):
                for j in range(i + 1, len(cited_sub)):
                    ri = cited_sub.iloc[i]
                    rj = cited_sub.iloc[j]
                    shared = 0

                    ai = set([a.strip().lower() for a in re.split(r";|,", str(ri["authors"])) if a.strip()])
                    aj = set([a.strip().lower() for a in re.split(r";|,", str(rj["authors"])) if a.strip()])
                    ki = set([k.strip().lower() for k in re.split(r";|,", str(ri["keywords"])) if k.strip()])
                    kj = set([k.strip().lower() for k in re.split(r";|,", str(rj["keywords"])) if k.strip()])

                    if len(ai & aj) > 0:
                        shared += 1
                    if str(ri["source"]).strip().lower() == str(rj["source"]).strip().lower() and str(ri["source"]).strip():
                        shared += 1
                    if len(ki & kj) > 0:
                        shared += 1

                    if shared >= citation_threshold:
                        Gc.add_edge(str(ri["title"])[:60], str(rj["title"])[:60], weight=shared)

            if Gc.number_of_edges() > 0:
                fig_cit, ax_cit = plt.subplots(figsize=(10, 10))
                pos = nx.spring_layout(Gc, seed=42)
                sizes = [50 + 8 * Gc.nodes[n]["citations"] for n in Gc.nodes()]
                nx.draw(Gc, pos, with_labels=True, node_size=sizes, font_size=7, ax=ax_cit)
                ax_cit.set_title("Citation-related network")
                plt.tight_layout()
                st.pyplot(fig_cit)
                figures["citation_network.png"] = fig_cit

                citation_edges = pd.DataFrame(
                    [{"paper1": u, "paper2": v, "weight": d["weight"]} for u, v, d in Gc.edges(data=True)]
                )
                sheets["citation_network_edges"] = citation_edges

    with tab4:
        st.subheader("Advanced meta-analysis")

        meta_source = filtered_master.copy()
        meta_source[["effect_size", "lower_ci", "upper_ci"]] = meta_source["abstract"].apply(extract_effect_info)

        for col in ["effect_size", "lower_ci", "upper_ci"]:
            meta_source[col] = pd.to_numeric(meta_source[col], errors="coerce")

        st.markdown("### Manual effect-size editor")
        editor_cols = ["title", "source", "year", "effect_size", "lower_ci", "upper_ci"]
        editable_df = meta_source[editor_cols].copy()
        editable_df["effect_size"] = pd.to_numeric(editable_df["effect_size"], errors="coerce")
        editable_df["lower_ci"] = pd.to_numeric(editable_df["lower_ci"], errors="coerce")
        editable_df["upper_ci"] = pd.to_numeric(editable_df["upper_ci"], errors="coerce")

        edited_df = st.data_editor(
            editable_df,
            num_rows="dynamic",
            use_container_width=True,
            key="effect_editor"
        )

        edited_df["effect_size"] = pd.to_numeric(edited_df["effect_size"], errors="coerce")
        edited_df["lower_ci"] = pd.to_numeric(edited_df["lower_ci"], errors="coerce")
        edited_df["upper_ci"] = pd.to_numeric(edited_df["upper_ci"], errors="coerce")
        sheets["manual_effect_editor"] = edited_df

        meta_df = edited_df.dropna(subset=["effect_size", "lower_ci", "upper_ci"]).copy()
        meta_df = meta_df[
            (meta_df["effect_size"] > 0) &
            (meta_df["lower_ci"] > 0) &
            (meta_df["upper_ci"] > 0) &
            (meta_df["lower_ci"] <= meta_df["effect_size"]) &
            (meta_df["effect_size"] <= meta_df["upper_ci"])
        ].copy()

        st.metric("Studies available for meta-analysis", len(meta_df))

        if len(meta_df) == 0:
            st.warning("No valid effect sizes are available after editing.")
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
                pooled_log_fe, pooled_se_fe, w_fe, z_fe, p_fe = fixed_effect_pool(meta_df)
                pooled_fe, pooled_fe_low, pooled_fe_high = summarize_pool(pooled_log_fe, pooled_se_fe)

                pooled_log_re, pooled_se_re, w_re, Q, df_q, I2, H2, tau2, z_re, p_re = random_effect_pool(meta_df)
                pooled_re, pooled_re_low, pooled_re_high = summarize_pool(pooled_log_re, pooled_se_re)
                pred_low, pred_high = prediction_interval_random(pooled_log_re, tau2)

                meta_df["weight_fixed"] = w_fe
                meta_df["weight_random"] = w_re
                meta_df["weight_fixed_pct"] = 100 * meta_df["weight_fixed"] / meta_df["weight_fixed"].sum()
                meta_df["weight_random_pct"] = 100 * meta_df["weight_random"] / meta_df["weight_random"].sum()

                summary_df = pd.DataFrame([{
                    "n_studies": len(meta_df),
                    "fixed_effect": pooled_fe,
                    "fixed_low": pooled_fe_low,
                    "fixed_high": pooled_fe_high,
                    "fixed_se": pooled_se_fe,
                    "fixed_z": z_fe,
                    "fixed_p": p_fe,
                    "random_effect": pooled_re,
                    "random_low": pooled_re_low,
                    "random_high": pooled_re_high,
                    "random_se": pooled_se_re,
                    "random_z": z_re,
                    "random_p": p_re,
                    "Q": Q,
                    "df": df_q,
                    "I2_percent": I2,
                    "H2": H2,
                    "tau2": tau2,
                    "prediction_low": pred_low,
                    "prediction_high": pred_high
                }])

                sheets["meta_analysis_ready"] = meta_df
                sheets["meta_summary"] = summary_df

                st.markdown("### Overall pooled estimates")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Studies", len(meta_df))
                c2.metric("Fixed effect", f"{pooled_fe:.3f}")
                c3.metric("Random effects", f"{pooled_re:.3f}")
                c4.metric("I²", f"{I2:.2f}%")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Q", f"{Q:.3f}")
                c6.metric("Tau²", f"{tau2:.4f}")
                c7.metric("H²", f"{H2:.3f}" if pd.notna(H2) else "NA")
                c8.metric("Prediction interval", f"{pred_low:.3f} to {pred_high:.3f}")

                st.dataframe(summary_df, use_container_width=True)

                weight_table = meta_df[[
                    "title", "source", "year", "effect_size", "lower_ci", "upper_ci",
                    "weight_fixed_pct", "weight_random_pct"
                ]].copy().sort_values("weight_random_pct", ascending=False)
                sheets["meta_weights"] = weight_table

                st.markdown("### Study weights")
                st.dataframe(weight_table, use_container_width=True)

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

                    egger_intercept = egger_model.intercept_
                    egger_slope = egger_model.coef_[0]

                    egger_summary = pd.DataFrame([{
                        "egger_intercept": egger_intercept,
                        "egger_slope": egger_slope,
                        "n_studies": len(egger_df)
                    }])
                    sheets["egger_summary"] = egger_summary

                    st.markdown("### Small-study bias check")
                    st.dataframe(egger_summary, use_container_width=True)

                    fig_egger, ax_egger = plt.subplots(figsize=(6, 5))
                    ax_egger.scatter(egger_df["precision"], egger_df["std_effect"])
                    x_line = np.linspace(egger_df["precision"].min(), egger_df["precision"].max(), 100)
                    y_line = egger_intercept + egger_slope * x_line
                    ax_egger.plot(x_line, y_line)
                    ax_egger.set_xlabel("Precision (1/SE)")
                    ax_egger.set_ylabel("Standardized effect")
                    ax_egger.set_title("Egger-style regression")
                    plt.tight_layout()
                    st.pyplot(fig_egger)
                    figures["egger_regression.png"] = fig_egger

                loo_rows = []
                for idx in meta_df.index:
                    loo_df = meta_df.drop(index=idx)
                    if len(loo_df) < 2:
                        continue
                    loo_log, loo_se, _, _, _, loo_I2, loo_H2, loo_tau2, loo_z, loo_p = random_effect_pool(loo_df)
                    loo_eff, loo_low, loo_high = summarize_pool(loo_log, loo_se)
                    loo_rows.append({
                        "removed_study": meta_df.loc[idx, "title"],
                        "pooled_random": loo_eff,
                        "low": loo_low,
                        "high": loo_high,
                        "I2_percent": loo_I2,
                        "tau2": loo_tau2
                    })

                loo_table = pd.DataFrame(loo_rows)
                if len(loo_table) > 0:
                    loo_table = loo_table.sort_values("pooled_random")
                    sheets["leave_one_out"] = loo_table
                    st.markdown("### Leave-one-out sensitivity analysis")
                    st.dataframe(loo_table, use_container_width=True)

                cum_df = meta_df.dropna(subset=["year_num"]).sort_values("year_num").reset_index(drop=True)
                cumulative_rows = []
                if len(cum_df) >= 2:
                    for i in range(2, len(cum_df) + 1):
                        sub = cum_df.iloc[:i].copy()
                        c_log, c_se, _, c_Q, c_dfq, c_I2, c_H2, c_tau2, c_z, c_p = random_effect_pool(sub)
                        c_eff, c_low, c_high = summarize_pool(c_log, c_se)
                        cumulative_rows.append({
                            "k": i,
                            "last_year": int(sub["year_num"].max()),
                            "pooled_random": c_eff,
                            "low": c_low,
                            "high": c_high,
                            "I2_percent": c_I2
                        })

                    cumulative_table = pd.DataFrame(cumulative_rows)
                    sheets["cumulative_meta"] = cumulative_table
                    st.markdown("### Cumulative meta-analysis by year")
                    st.dataframe(cumulative_table, use_container_width=True)

                st.markdown("### Subgroup meta-analysis")
                subgroup_option = st.selectbox("Choose subgroup variable", ["source", "year_group"], key="subgroup_select")

                subgroup_df = meta_df.copy()
                median_year = subgroup_df["year_num"].dropna().median() if subgroup_df["year_num"].notna().any() else np.nan
                subgroup_df["year_group"] = np.where(
                    subgroup_df["year_num"].fillna(-9999) < median_year,
                    "Earlier",
                    "Later"
                )

                subgroup_rows = []
                for grp, sub in subgroup_df.groupby(subgroup_option):
                    if len(sub) < 2:
                        continue
                    s_log, s_se, _, s_Q, s_dfq, s_I2, s_H2, s_tau2, s_z, s_p = random_effect_pool(sub)
                    s_eff, s_low, s_high = summarize_pool(s_log, s_se)
                    subgroup_rows.append({
                        "subgroup": grp,
                        "n_studies": len(sub),
                        "pooled_random": s_eff,
                        "low": s_low,
                        "high": s_high,
                        "z": s_z,
                        "p": s_p,
                        "I2_percent": s_I2,
                        "tau2": s_tau2
                    })

                subgroup_table = pd.DataFrame(subgroup_rows)
                if len(subgroup_table) > 0:
                    sheets["subgroup_meta"] = subgroup_table
                    st.dataframe(subgroup_table, use_container_width=True)

                st.markdown("### Meta-regression by year")
                reg_df = meta_df.dropna(subset=["year_num"]).copy()
                if len(reg_df) >= 3:
                    X_reg = reg_df[["year_num"]].values
                    y_reg = reg_df["log_effect"].values
                    weights_reg = 1 / (reg_df["se"] ** 2)

                    reg_model = LinearRegression()
                    reg_model.fit(X_reg, y_reg, sample_weight=weights_reg)

                    reg_pred = reg_model.predict(X_reg)
                    meta_reg_df = pd.DataFrame([{
                        "intercept": reg_model.intercept_,
                        "slope_year": reg_model.coef_[0],
                        "n_studies": len(reg_df)
                    }])
                    sheets["meta_regression_year"] = meta_reg_df
                    st.dataframe(meta_reg_df, use_container_width=True)

                    fig_reg, ax_reg = plt.subplots(figsize=(7, 5))
                    ax_reg.scatter(reg_df["year_num"], reg_df["log_effect"], s=40 + 4 * weights_reg)
                    order = np.argsort(reg_df["year_num"].values)
                    ax_reg.plot(reg_df["year_num"].values[order], reg_pred[order])
                    ax_reg.set_xlabel("Year")
                    ax_reg.set_ylabel("Log effect size")
                    ax_reg.set_title("Meta-regression by year")
                    plt.tight_layout()
                    st.pyplot(fig_reg)
                    figures["meta_regression_year.png"] = fig_reg
                else:
                    st.info("At least 3 studies with valid year are needed for meta-regression.")

                influence_rows = []
                for idx in meta_df.index:
                    contrib_q = meta_df.loc[idx, "weight_fixed"] * ((meta_df.loc[idx, "log_effect"] - pooled_log_fe) ** 2)
                    diff_in_pooled = abs(meta_df.loc[idx, "log_effect"] - pooled_log_re)
                    influence_rows.append({
                        "study": meta_df.loc[idx, "title"],
                        "contribution_to_Q": contrib_q,
                        "distance_from_pooled_log": diff_in_pooled,
                        "weight_random_pct": meta_df.loc[idx, "weight_random_pct"]
                    })

                influence_table = pd.DataFrame(influence_rows).sort_values(
                    ["contribution_to_Q", "distance_from_pooled_log"],
                    ascending=False
                )
                sheets["influence_table"] = influence_table
                st.markdown("### Influence diagnostics")
                st.dataframe(influence_table, use_container_width=True)

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
