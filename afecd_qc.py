# afecd_qc.py
# Quick debug / QC / QA for AFECD parsed CSV
import argparse
import os
import sys
import re
import pandas as pd

REQ_COLS = [
    "row_type","section","section_index","section_label","section_sort",
    "specialty_title","service","major_group_code","major_group_name",
    "afsc_family","afsc_code","afsc_code_plain",
    "skill_level_num","skill_level_name",
    "text",
    "page_start","page_end","effective_date","source_file",
    "search_key","join_key_afsc_skill","id"
]

def read_csv_safely(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")

def ensure_cols(df):
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        print(f"[WARN] CSV missing columns: {missing}")
        for c in missing: df[c] = ""
    return df

def write_slice(df, outdir, name):
    if df.empty: return
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{name}.csv")
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"  -> wrote {len(df)} rows: {out}")

def main():
    ap = argparse.ArgumentParser(description="AFECD CSV QC/QA")
    ap.add_argument("--input", required=True, help="Path to parsed CSV")
    ap.add_argument("--outdir", default="./afecd_qc_out", help="Directory for problem slices")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERR] Input not found: {args.input}")
        sys.exit(1)

    df = read_csv_safely(args.input)
    df = ensure_cols(df)

    N = len(df)
    print(f"\nLoaded {N} rows from {args.input}")

    # Basic counts
    print("\nRow counts by type:")
    print(df["row_type"].value_counts(dropna=False).to_string())

    # Duplicate IDs
    dup = df[df.duplicated("id", keep=False)].sort_values("id")
    print(f"\nDuplicate id rows: {len(dup)}")
    write_slice(dup, args.outdir, "duplicate_ids")

    # Empty or super-short text
    empty_text = df[df["text"].astype(str).str.strip().eq("")]
    print(f"Empty text rows: {len(empty_text)}")
    write_slice(empty_text, args.outdir, "empty_text")

    # Weird characters (e.g., lingering star symbol)
    weird_mask = df["text"].astype(str).str.contains("[\u2022\u25CF\u2605\u2606\u272A\uF0B7\uF0A7\uF0D8\uF0FC\u2020\u00AD\uF0B2\uF0A3\uF0A9\uF0BD\uF0B4\uF0B1\uF0D0\uF0E0\uF0E7\uF0E8\uF0E9\uF0EA\uF0EB\uF0EC\uF0ED\uF0EE\uF0EF\uF0F0\uF0F1\uF0F2\uF0F3\uF0F4\uF0F5\uF0F6\uF0F7\uF0F8\uF0F9\uF0FA\uF0FB\uF0FD\uF0FE\uF0FF\uF000\uF001\uF002\u25A0\u25A1\u25AA\u25AB\u25B6\u25C6\u25C7\u25CF\u25CB\u2666\u2665\u2663\u2660\u0E3F\u2021\u2026\u2060\uFEFF]", regex=True)
    weird = df[weird_mask]
    print(f"Weird-char rows (potential cleanup): {len(weird)}")
    write_slice(weird, args.outdir, "weird_chars")

    # CEM (00) leak check
    cem = df[df["skill_level_num"].astype(str).eq("00")]
    print(f"CEM '00' rows (should be 0 for AFSC fanout): {len(cem)}")
    write_slice(cem, args.outdir, "cem_leaks")

    # AFSC vs SDI split
    # SDI if: skill_level_num is "" or "SDI", OR the code looks like an SDI (e.g., 8A200)
    is_sdi = (
        df["skill_level_num"].astype(str).str.upper().isin(["", "SDI"]) |
        df["afsc_code_plain"].astype(str).str.fullmatch(r"[0-9A-Z]{2}\d{3}", na=False)
    )
    sdi_df  = df[is_sdi].copy()
    afsc_df = df[~is_sdi].copy()
    print(f"\nAFSC rows: {len(afsc_df)}   SDI rows: {len(sdi_df)}")

    # --- AFSC QA ---
    if not afsc_df.empty:
        # Must be in {3,5,7,9} for true AFSCs only
        bad_levels = afsc_df[~afsc_df["skill_level_num"].astype(str).isin(["3","5","7","9"])]
        print(f"AFSC rows with unexpected skill_level_num: {len(bad_levels)}")
        write_slice(bad_levels, args.outdir, "afsc_bad_levels")

        # Missing summary or duty per AFSC+level
        matrix = (afsc_df
                  .pivot_table(index=["afsc_code_plain","skill_level_num"],
                               columns="row_type", values="id",
                               aggfunc="count", fill_value=0))
        # Ensure both 'summary' and 'duty' columns exist for robust query
        for col in ["summary", "duty"]:
            if col not in matrix.columns:
                matrix[col] = 0
        miss = matrix.query("(summary == 0) or (duty == 0)").reset_index()
        print(f"AFSC ladders missing summary/duty: {len(miss)}")
        write_slice(miss, args.outdir, "afsc_missing_summary_or_duty")

        # Experience rows sanity
        exp = afsc_df[afsc_df["row_type"].eq("experience")]
        bad_exp_idx = exp[~exp["section_index"].astype(str).str.match(r"^3\.\d+\.\d+$")]
        print(f"Experience rows with odd section_index: {len(bad_exp_idx)}")
        write_slice(bad_exp_idx, args.outdir, "experience_bad_section_index")

    # --- SDI QA ---
    if not sdi_df.empty:
        # SDI should have blank major_group_code/afsc_family/skill_level_num
        sdi_bad_fields = sdi_df[
            sdi_df["major_group_code"].astype(str).str.strip().ne("") |
            sdi_df["afsc_family"].astype(str).str.strip().ne("") |
            sdi_df["skill_level_num"].astype(str).str.strip().ne("")
        ]
        print(f"SDI with unexpected AFSC fields populated: {len(sdi_bad_fields)}")
        write_slice(sdi_bad_fields, args.outdir, "sdi_bad_fields")

        # SDI completeness (summary + at least one duty)
        sdi_matrix = (sdi_df
                      .assign(dummy=1)
                      .pivot_table(index=["afsc_code_plain"], columns="row_type",
                                   values="dummy", aggfunc="count", fill_value=0))
        # Ensure both 'summary' and 'duty' columns exist for robust query
        for col in ["summary", "duty"]:
            if col not in sdi_matrix.columns:
                sdi_matrix[col] = 0
        sdi_miss = sdi_matrix.query("(summary == 0) or (duty == 0)").reset_index()
        print(f"SDIs missing summary or duty: {len(sdi_miss)}")
        write_slice(sdi_miss, args.outdir, "sdi_missing_summary_or_duty")

        # SDI titles that don't start with "SDI <code>"
        bad_sdi_titles = sdi_df[~sdi_df["specialty_title"].astype(str).str.contains(r"\bSDI\s+[0-9A-Z]{2}\d{3}\b", regex=True)]
        print(f"SDI rows with title not containing 'SDI <code>': {len(bad_sdi_titles)}")
        write_slice(bad_sdi_titles, args.outdir, "sdi_bad_title_format")

    # Page ranges sanity
    try:
        ps = pd.to_numeric(df["page_start"], errors="coerce")
        pe = pd.to_numeric(df["page_end"], errors="coerce")
        bad_pages = df[ (ps.isna()) | (pe.isna()) | (ps > pe) ]
        print(f"Rows with bad page range: {len(bad_pages)}")
        write_slice(bad_pages, args.outdir, "bad_page_ranges")
    except Exception as e:
        print(f"[WARN] page_start/page_end check skipped: {e}")

    # Titles that look suspiciously short or split (e.g., 'B G', 'B I')
    def looks_split_title(t):
        toks = re.findall(r"[A-Z]+", str(t).strip())
        return 0 < len(toks) <= 2 and all(len(x) <= 2 for x in toks)
    bad_titles = df[df["specialty_title"].apply(looks_split_title)]
    print(f"Suspicious titles (possibly split lines): {len(bad_titles)}")
    write_slice(bad_titles, args.outdir, "suspicious_titles")

    # Top titles by row count (quick glance)
    print("\nTop specialty_title by rows:")
    top_titles = (df["specialty_title"]
                  .value_counts()
                  .head(15))
    print(top_titles.to_string())

    print("\nQC complete.")

if __name__ == "__main__":
    main()
