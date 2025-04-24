import os
import json
import math
import random
import argparse

import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

# -----------------------------------------------------------------------------
# Part 1: Convert only test_*.tables.jsonl → txt chunks + multiple formats
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Part 3: Sentence-similarity benchmarking
# -----------------------------------------------------------------------------
# load models/scorers once
_MODEL   = SentenceTransformer("all-MiniLM-L6-v2")
_SCORER  = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)

def compare_sentences(hyps, refs):
    """
    Given two equally-long lists of generated (hyps) and reference (refs)
    sentences, returns for each pair a dict with:
      - BLEU
      - ROUGE1/2/L f-measures
      - embedding cosine similarity
    """
    assert len(hyps) == len(refs), "Length mismatch"

    # embeddings
    emb_h = _MODEL.encode(hyps, convert_to_tensor=True)
    emb_r = _MODEL.encode(refs, convert_to_tensor=True)
    cosines = util.pytorch_cos_sim(emb_h, emb_r).diag().tolist()

    results = []
    for i, (h, r) in enumerate(zip(hyps, refs)):
        bleu  = sentence_bleu([r.split()], h.split())
        rouge = _SCORER.score(r, h)
        results.append({
            "hypothesis":      h,
            "reference":       r,
            "bleu":            bleu,
            "rouge1":          rouge["rouge1"].fmeasure,
            "rouge2":          rouge["rouge2"].fmeasure,
            "rougeL":          rouge["rougeL"].fmeasure,
            "cosine_similarity": cosines[i],
        })
    return results

# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert .tables.jsonl → multiple formats + sample test_*.txt"
    )
    parser.add_argument("--input_dir",  required=True, help="dir with .tables.jsonl files")
    parser.add_argument("--output_dir", required=True, help="where to save converted files")
    args = parser.parse_args()

    convert_test_tables_to_formats(args.input_dir, args.output_dir)

    # Example usage of compare_sentences():
    # hyps = ["This is a test.", "Another sentence."]
    # refs = ["This is a test!", "Another line."]
    # scores = compare_sentences(hyps, refs)
    # print(pd.DataFrame(scores).round(3))
