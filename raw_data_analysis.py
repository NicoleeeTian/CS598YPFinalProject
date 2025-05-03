import os
import math
import json
import random
from pathlib import Path
from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

def convert_test_tables_to_formats(input_dir, output_dir):
    """
    1) Finds files starting with 'test' and ending in '.tables.jsonl' in input_dir.
    2) For each file, loads all tables and segments them into up to 50 text chunks.
    3) Randomly selects 5 chunk indices.
    4) For each selected chunk:
       • Writes the chunk as a .txt file in 'txt/'
       • Converts the underlying tables in that chunk into CSV, Markdown, HTML, XLSX, and JSON
         and saves them into their respective subfolders under output_dir.
    """
    # create subfolders for each format
    dirs = {
        'txt': os.path.join(output_dir, 'txt'),
        'csv': os.path.join(output_dir, 'csv'),
        'md': os.path.join(output_dir, 'md'),
        'html': os.path.join(output_dir, 'html'),
        'xlsx': os.path.join(output_dir, 'xlsx'),
        'json': os.path.join(output_dir, 'json'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # process test*.tables.jsonl files
    for fname in os.listdir(input_dir):
        if not (fname.startswith("test") and fname.endswith(".tables.jsonl")):
            continue

        stem = fname[:-len(".tables.jsonl")]
        path = os.path.join(input_dir, fname)

        # Load all tables
        tables = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    tables.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON in {fname}")

        if not tables:
            continue

        # Build entries for each table
        entries = []  # list of (table_obj, text_repr)
        for tbl in tables:
            header = tbl.get("header", [])
            rows = tbl.get("rows", [])
            parts = [
                f"=== Table ID: {tbl.get('id','?')} ===\n",
                f"=== Table Title: {tbl.get('page_title','N/A')} ===\n",
                "Columns: " + ", ".join(header) + "\n"
            ]
            for row in rows:
                parts.append(f"  - {', '.join(map(str, row))}\n")
            parts.append("\n")
            entries.append((tbl, "".join(parts)))

        total = len(entries)
        num_chunks = min(50, total)
        chunk_size = math.ceil(total / num_chunks)

        # pick 5 random chunk indices
        selected = random.sample(range(num_chunks), min(5, num_chunks))

        for idx in selected:
            start = idx * chunk_size
            end = start + chunk_size
            segment = entries[start:end]
            if not segment:
                continue

            # write txt chunk
            txt_out = os.path.join(dirs['txt'], f"{stem}_chunk_{idx}.txt")
            with open(txt_out, 'w', encoding='utf-8') as ftxt:
                for _, txt in segment:
                    ftxt.write(txt)

            # convert each table in segment
            for tbl, _ in segment:
                tid = tbl.get("id","")
                base = f"{stem}_chunk_{idx}__{tid}"
                df = pd.DataFrame(tbl.get("rows", []), columns=tbl.get("header", []))

                # CSV
                df.to_csv(os.path.join(dirs['csv'], base + '.csv'), index=False)
                # Markdown
                with open(os.path.join(dirs['md'], base + '.md'), 'w', encoding='utf-8') as f_md:
                    f_md.write(df.to_markdown(index=False))
                # HTML
                df.to_html(os.path.join(dirs['html'], base + '.html'), index=False)
                # Excel
                df.to_excel(os.path.join(dirs['xlsx'], base + '.xlsx'), index=False)
                # JSON
                with open(os.path.join(dirs['json'], base + '.json'), 'w', encoding='utf-8') as f_json:
                    json.dump(tbl, f_json, ensure_ascii=False, indent=2)

        print(f"✅ {fname}: created chunks {sorted(selected)} in subfolders.")


def run_similarity_on_file(input_csv, output_csv):
    # Initialize models
    _MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    _SCORER = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    _SENTIMENT_ANALYZER = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def clean_text(text):
        return str(text).strip()

    def analyze_sentiment(text):
        result = _SENTIMENT_ANALYZER(text[:512])[0]
        return result['label'].lower()

    def missing_info_ratio(hyp, ref):
        hyp_words = set(str(hyp).lower().split())
        ref_words = set(str(ref).lower().split())
        missing = ref_words - hyp_words
        return len(missing) / (len(ref_words) + 1e-8)

    df = pd.read_csv(input_csv)
    df["Name"] = df["Name"].apply(clean_text)
    df["Query"] = df["Query"].apply(clean_text)
    df["Type"] = df["Type"].apply(clean_text)
    df["Answer"] = df["Answer"].apply(clean_text)

    # Group by Name and Query
    grouped = df.groupby(["Name", "Query"])
    valid_groups = grouped.filter(lambda x: x["Type"].nunique() == 3)
    final_groups = valid_groups.groupby(["Name", "Query"])

    results = []

    for (name, query), group in final_groups:
        group = group.reset_index(drop=True)
        entries = list(group[["Answer", "Type"]].itertuples(index=False, name=None))
        
        for (ans1, type1), (ans2, type2) in combinations(entries, 2):
            emb1 = _MODEL.encode(ans1, convert_to_tensor=True)
            emb2 = _MODEL.encode(ans2, convert_to_tensor=True)
            cosine = util.pytorch_cos_sim(emb1, emb2).item()
            
            rouge = _SCORER.score(ans2, ans1)
            sentiment1 = analyze_sentiment(ans1)
            sentiment2 = analyze_sentiment(ans2)
            missing_ratio = missing_info_ratio(ans1, ans2)

            results.append({
                "Name": name,
                "Query": query,
                "Answer1": ans1,
                "Answer2": ans2,
                "Type1": type1,
                "Type2": type2,
                "TypePair": " + ".join(sorted([type1, type2])),
                "Cosine_similarity": cosine,
                "Answer1_sentiment": sentiment1,
                "Answer2_sentiment": sentiment2,
                "Rouge1": rouge["rouge1"].fmeasure,
                "Rouge2": rouge["rouge2"].fmeasure,
                "RougeL": rouge["rougeL"].fmeasure,
                "MissingInfoRatio": missing_ratio,
                "Answer1_Length": len(ans1.split()),
                "Answer2_Length": len(ans2.split()),
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Saved results to: {output_csv}")


def split_by_chunks(input_path: str) -> List[str]:
    """
    Split *input_path* into separate CSVs, one for each unique
    `NoOfChunksRetrieved` value. Returns the paths of the new files.
    """
    print(input_path)
    df = pd.read_csv(input_path)
    out_files: List[str] = []

    for chunks, grp in df.groupby("NoOfChunksRetrieved", sort=False):
        out_file = f"{Path(input_path).stem}_chunks_{chunks}.csv"
        grp.to_csv(out_file, index=False)
        out_files.append(out_file)

    return out_files


def correctness_percentage_by_type(
    input_path: str,
    correctness_col: str = "Correctness",
    type_col: str = "Type",
) -> pd.DataFrame:

    # 1. Load data
    df = pd.read_csv(input_path)

    # 2. Normalise the correctness flag → Boolean
    df[correctness_col] = (
        df[correctness_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({"TRUE": True, "FALSE": False})
    )

    # (Optional) Drop rows where correctness is missing / unrecognised
    df = df.dropna(subset=[correctness_col])

    # 3. Group by type and compute percentage of TRUE
    pct_series = df.groupby(type_col)[correctness_col].mean() * 100

    # 4. Return a clean DataFrame
    return (
        pct_series.reset_index()
        .rename(columns={correctness_col: "pct_correct"})
        .sort_values("pct_correct", ascending=False, ignore_index=True)
    )


def hit_rates(csv_path: str,
              name_col: str = "Name",
              docs_col: str = "RetrievedDocs",
              type_col: str = "Type",
              doc_sep: str = ",") -> None:
    df = pd.read_csv(csv_path)

    def name_in_docs(row) -> bool:
        docs = [doc.strip() for doc in str(row[docs_col]).split(doc_sep)]
        str_docs = ""
        for d in docs:
            str_docs += d + " "
        # print(row[name_col], docs)
        # print(str(row[name_col]).strip(), str_docs)
        return str(row[name_col]).strip() in str_docs

    df["hit"] = df.apply(name_in_docs, axis=1)
    overall_pct = 100 * df["hit"].mean()
    type_pct = 100 * df.groupby(type_col)["hit"].mean().sort_index()

    print(f"OVERALL hit-rate: {overall_pct:.2f}%  "
          f"({df['hit'].sum()} of {len(df)} rows)\n")

    print("Retrieved Documents Hit-rate by type:")
    for t, p in type_pct.items():
        n = df[df[type_col] == t].shape[0]
        k = df[(df[type_col] == t) & df["hit"]].shape[0]
        print(f"  {t}: {p:.2f}%  ({k} of {n})")
    
if __name__ == "__main__":
    print(correctness_percentage_by_type("CS598YP-FinalProject_chunks_1.csv"))
    print(correctness_percentage_by_type("CS598YP-FinalProject_chunks_12.csv"))
    print(hit_rates("CS598YP-FinalProject_chunks_12.csv"))

    run_similarity_on_file(
        "CS598YP-FinalProject_chunks_1.csv",
        "group_similarity_chunks_1.csv")
    run_similarity_on_file(
        "CS598YP-FinalProject_chunks_12.csv",
        "group_similarity_chunks_12.csv")