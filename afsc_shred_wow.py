# pip install pdfplumber pandas pyarrow
import re, json, hashlib
from pathlib import Path
import pdfplumber
import pandas as pd

PDF_PATH = r"C:\Users\krced\EnlistedClass\enlistedclasspdf\EnlClassStructue.pdf"
OUT_DIR  = Path(r"C:\Users\krced\EnlistedClass\afsc_schred_to_vectordb")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_BASE = OUT_DIR / "afsc_catalog_enriched"   # writes .csv/.parquet/.jsonl

# ============ helpers ============
def norm_ws(s: str) -> str:
    s = "".join(" " if ch.isspace() else ch for ch in (s or ""))
    s = s.replace("\u200b","").replace("\u200c","").replace("\u200d","").replace("\ufeff","").replace("\u2060","")
    return re.sub(r"\s+", " ", s).strip()

# 2A355, 2A3X5, 1A100, 1A132*
AFSC_RE       = re.compile(r"^\d[A-Z]\d(?:\d\d|X\d|00)\*?$")
AFSC_SCAN_RE  = re.compile(r"\b\d[A-Z]\d(?:\d\d|X\d|00)\*?\b")
MAJOR_RE      = re.compile(r"^([A-Z][A-Z ]+)\s+\((\d[A-Z])\)$")

skill_name = {
    "1":"Input/Helper", "3":"Apprentice/Semiskilled", "5":"Journeyman/Skilled",
    "7":"Craftsman/Advanced", "9":"Superintendent", "00":"Chief Enlisted Manager"
}
skill_tier = {"1":"apprentice","3":"apprentice","5":"journeyman","7":"craftsman","9":"superintendent","00":"cem"}

def words_by_lines(page, y_tol=3):
    words = page.extract_words(use_text_flow=True, keep_blank_chars=False, x_tolerance=2, y_tolerance=y_tol)
    lines = {}
    for w in words:
        y = round(w["top"], 0)
        lines.setdefault(y, []).append(w)
    out = []
    for y in sorted(lines):
        ws = sorted(lines[y], key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in ws)
        out.append((y, text, ws))
    return out

def read_code(tokens, i):
    # Assemble an AFSC from up to 4 tokens (+ optional '*' next)
    for k in (1,2,3,4):
        if i+k > len(tokens): break
        cand = "".join(tokens[i:i+k])
        if AFSC_RE.fullmatch(cand): return cand, i+k
        if i+k < len(tokens) and tokens[i+k] == "*" and AFSC_RE.fullmatch(cand+"*"):
            return cand+"*", i+k+1
    return None, i+1

def scan_shreds_tokens(tokens):
    # Single-letter tokens mark shreds; title runs until the next single-letter
    out, i = [], 0
    while i < len(tokens):
        tok = tokens[i]
        if re.fullmatch(r"[A-Z]", tok):
            letter = tok
            j = i+1
            parts = []
            while j < len(tokens) and not re.fullmatch(r"[A-Z]", tokens[j]):
                parts.append(tokens[j]); j += 1
            title = norm_ws(" ".join(parts))
            if title:
                out.append((letter, title))
            i = j
        else:
            i += 1
    # de-dupe within line
    seen, clean = set(), []
    for p in out:
        if p in seen: continue
        seen.add(p); clean.append(p)
    return clean

def skill_from_code(code: str) -> str:
    # strip '*' ; if endswith '00' -> '00' ; else 4th char (5th if 4th=='X')
    c = str(code).strip().rstrip("*")
    if len(c) < 5: return ""
    if c.endswith("00"): return "00"
    return c[4] if c[3] == "X" else c[3]

def aircraft_tags_from(title: str) -> str:
    # A-10, F-16, C-130J, E-3G, KC-46, EC-130H, EA-37B
    models = re.findall(r"\b([A-Z]{1,2}-?\d{1,3}[A-Z]?)\b", title or "")
    return ",".join(sorted(set(models)))

def make_id(service, mg_code, code, lvl, shred_code, shred_title):
    import hashlib
    key = "|".join([service, mg_code or "", code, str(lvl), shred_code or "", shred_title or ""])
    return f"{code.lower()}:{lvl}:{(shred_code or '_').lower()}:{hashlib.sha1(key.encode()).hexdigest()[:10]}"

def ensure_code_seen(code, service, major_code, major_name, page_number, seen_afscs):
    if code not in seen_afscs:
        seen_afscs[code] = {
            "service": service,
            "major_group_code": major_code,
            "major_group_name": major_name,
            "afsc_title": "",      # may fill later
            "page": page_number,
        }

# ============ parse ============
rows = []
service = "USAF"
major_code = major_name = ""

pending_afscs = []            # [(code, title)]
pending_shreds = []           # [(letter, title)] if shreds show before AFSCs
current_code = None
current_title_parts = []
in_shreds_block = False
seen_afscs = {}               # code -> dict for base synthesis

with pdfplumber.open(PDF_PATH) as pdf:
    for p in pdf.pages:
        page_text = norm_ws(p.extract_text() or "")

        # register any codes visible in raw text (safety net)
        for raw_code in set(AFSC_SCAN_RE.findall(page_text)):
            ensure_code_seen(raw_code, service, major_code, major_name, p.page_number, seen_afscs)

        if "SPACE FORCE CLASSIFICATION STRUCTURE CHART" in page_text:
            service = "USSF"

        for _, line_text, ws in words_by_lines(p):
            t = norm_ws(line_text)
            tokens = [w["text"] for w in ws]

            # major group header
            mg = MAJOR_RE.match(t)
            if mg:
                if current_code and current_title_parts:
                    title = norm_ws(" ".join(current_title_parts))
                    pending_afscs.append((current_code, title))
                    ensure_code_seen(current_code, service, major_code, major_name, p.page_number, seen_afscs)
                    seen_afscs[current_code]["afsc_title"] = title
                pending_afscs, pending_shreds = [], []
                current_code, current_title_parts = None, []
                in_shreds_block = False
                major_name, major_code = mg.group(1).strip(), mg.group(2).strip()
                continue

            # 1) codes on this line
            i = 0
            found_code_on_line = False
            while i < len(tokens):
                code, j = read_code(tokens, i)
                if code:
                    found_code_on_line = True
                    ensure_code_seen(code, service, major_code, major_name, p.page_number, seen_afscs)

                    # finalize previous accumulating title
                    if current_code and current_title_parts:
                        title_prev = norm_ws(" ".join(current_title_parts))
                        pending_afscs.append((current_code, title_prev))
                        ensure_code_seen(current_code, service, major_code, major_name, p.page_number, seen_afscs)
                        seen_afscs[current_code]["afsc_title"] = title_prev
                        current_code, current_title_parts = None, []

                    # same-line title until next code
                    parts, k = [], j
                    while k < len(tokens):
                        nxt, _ = read_code(tokens, k)
                        if nxt: break
                        parts.append(tokens[k]); k += 1
                    title = norm_ws(" ".join(parts))
                    if title:
                        pending_afscs.append((code, title))
                        seen_afscs[code]["afsc_title"] = title
                    else:
                        current_code = code
                        current_title_parts = []
                    i = k
                else:
                    i = j

            if found_code_on_line:
                in_shreds_block = False
                # fan out any buffered shreds now
                if pending_shreds and pending_afscs:
                    for letter, s_title in pending_shreds:
                        for code, atitle in pending_afscs:
                            lvl = skill_from_code(code)
                            if lvl == "1":  # never include helpers
                                continue
                            rows.append({
                                "row_type": "shred",
                                "service": service,
                                "major_group_code": major_code,
                                "major_group_name": major_name,
                                "afsc_code": code,
                                "afsc_family": code[:3],
                                "skill_level_num": lvl,
                                "skill_level_name": skill_name.get(lvl, ""),
                                "skill_tier": skill_tier.get(lvl, ""),
                                "afsc_title": atitle,
                                "shred_code": letter,
                                "shred_title": s_title,
                                "aircraft_tags": aircraft_tags_from(s_title),
                                "authorized_without_shredouts": "true" if code.endswith("*") else "false",
                                "is_data_mask": "true" if (letter == "Z" and "mask" in s_title.lower()) else "false",
                                "page": str(p.page_number),
                                "effective_date": "2025-04-30",
                                "source_file": Path(PDF_PATH).name,
                            })
                pending_shreds = []
                continue

            # 2) shreds line
            shreds = scan_shreds_tokens(tokens)
            if shreds:
                if current_code and current_title_parts:
                    title = norm_ws(" ".join(current_title_parts))
                    pending_afscs.append((current_code, title))
                    ensure_code_seen(current_code, service, major_code, major_name, p.page_number, seen_afscs)
                    seen_afscs[current_code]["afsc_title"] = title
                    current_code, current_title_parts = None, []
                if pending_afscs:
                    for letter, s_title in shreds:
                        for code, atitle in pending_afscs:
                            lvl = skill_from_code(code)
                            if lvl == "1":
                                continue
                            rows.append({
                                "row_type": "shred",
                                "service": service,
                                "major_group_code": major_code,
                                "major_group_name": major_name,
                                "afsc_code": code,
                                "afsc_family": code[:3],
                                "skill_level_num": lvl,
                                "skill_level_name": skill_name.get(lvl, ""),
                                "skill_tier": skill_tier.get(lvl, ""),
                                "afsc_title": atitle,
                                "shred_code": letter,
                                "shred_title": s_title,
                                "aircraft_tags": aircraft_tags_from(s_title),
                                "authorized_without_shredouts": "true" if code.endswith("*") else "false",
                                "is_data_mask": "true" if (letter == "Z" and "mask" in s_title.lower()) else "false",
                                "page": str(p.page_number),
                                "effective_date": "2025-04-30",
                                "source_file": Path(PDF_PATH).name,
                            })
                    in_shreds_block = True
                else:
                    pending_shreds.extend(shreds)
                    in_shreds_block = True
                continue

            # 3) title continuation
            if current_code:
                current_title_parts.extend(tokens)
                continue

            # 4) end shreds block
            if in_shreds_block and not shreds:
                pending_afscs = []
                in_shreds_block = False

# ============ dataframe + base rows ============
df = pd.DataFrame(rows).drop_duplicates()

# base rows from every AFSC we saw (even if no shreds)
base_rows = []
for code, info in seen_afscs.items():
    lvl = skill_from_code(code)
    if lvl == "1":   # never include helpers
        continue
    base_rows.append({
        "row_type": "base",
        "service": info["service"],
        "major_group_code": info["major_group_code"],
        "major_group_name": info["major_group_name"],
        "afsc_code": code,
        "afsc_family": code[:3],
        "skill_level_num": lvl,
        "skill_level_name": skill_name.get(lvl, ""),
        "skill_tier": skill_tier.get(lvl, ""),
        "afsc_title": info["afsc_title"],   # may be blank; backfill below
        "shred_code": "",
        "shred_title": "",
        "aircraft_tags": "",
        "authorized_without_shredouts": "true" if code.endswith("*") else "false",
        "is_data_mask": "false",
        "page": str(info["page"]),
        "effective_date": "2025-04-30",
        "source_file": Path(PDF_PATH).name,
    })

df_base = pd.DataFrame(base_rows)
full = pd.concat([df, df_base], ignore_index=True)

# backfill blank base titles from any shred row of same AFSC
mask_base_blank = (full["row_type"]=="base") & (full["afsc_title"].fillna("").str.strip()=="")
if mask_base_blank.any():
    sh_title = (full[full["row_type"]=="shred"][["afsc_code","afsc_title"]]
                .dropna()
                .groupby("afsc_code")["afsc_title"]
                .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]))
    full.loc[mask_base_blank, "afsc_title"] = full.loc[mask_base_blank, "afsc_code"].map(sh_title).fillna("")

# easy-filter keys
full["afsc_code_plain"]  = full["afsc_code"].str.replace("*","", regex=False)
full["search_key_plain"] = full.apply(
    lambda r: f"{r['afsc_code_plain']}/{r['shred_code']}" if r["row_type"]=="shred" else r["afsc_code_plain"],
    axis=1
)
full["search_key"] = full.apply(
    lambda r: f"{r['afsc_code']}/{r['shred_code']}" if r["row_type"]=="shred" else r["afsc_code"],
    axis=1
)
full["join_key_afsc_skill"]       = full["afsc_code"] + "|" + full["skill_level_num"]
full["join_key_afsc_skill_shred"] = full["join_key_afsc_skill"] + "|" + full["shred_code"].fillna("")

# text + ids
def build_text(r):
    base = f"{r['service']} {r['major_group_code']} {r['afsc_code']} ({r['skill_level_name']}) : {r['afsc_title']}"
    return base if r["row_type"] == "base" else f"{base} | Shred {r['shred_code']}: {r['shred_title']}"
full["text"] = full.apply(build_text, axis=1)
full["id"] = full.apply(lambda r: make_id(r["service"], r["major_group_code"], r["afsc_code"],
                                          r["skill_level_num"], r["shred_code"], r["shred_title"]), axis=1)

# ======== canon & dedupe before writing ========
# never include helpers
full = full[full["skill_level_num"] != "1"].copy()

# 1) shreds: keep the longest title per AFSC+level+letter
sh = full[full["row_type"]=="shred"].copy()
if not sh.empty:
    sh["__len"] = sh["shred_title"].fillna("").str.len()
    sh = (sh.sort_values(["afsc_code_plain","skill_level_num","shred_code","__len"])
            .groupby(["afsc_code_plain","skill_level_num","shred_code"], as_index=False)
            .tail(1)
            .drop(columns="__len"))

# 2) base: keep exactly one per AFSC+level (prefer those with a title)
base = full[full["row_type"]=="base"].copy()
if not base.empty:
    base["__has_title"] = base["afsc_title"].fillna("").str.len()
    base = (base.sort_values(["afsc_code_plain","skill_level_num","__has_title"], ascending=[True, True, False])
                 .groupby(["afsc_code_plain","skill_level_num"], as_index=False)
                 .head(1)
                 .drop(columns="__has_title"))

# merge back
full = pd.concat([base, sh], ignore_index=True)
full = full.sort_values(["afsc_code_plain","skill_level_num","row_type","shred_code"]).reset_index(drop=True)

# rebuild text+id after canon (rows changed)
full["text"] = full.apply(build_text, axis=1)
full["id"] = full.apply(lambda r: make_id(r["service"], r["major_group_code"], r["afsc_code"],
                                          r["skill_level_num"], r["shred_code"], r["shred_title"]), axis=1)

# safety checks
assert not (full["skill_level_num"]=="1").any(), "Helper level leaked"
dupes = (full[full["row_type"]=="shred"]
         .groupby(["afsc_code_plain","skill_level_num","shred_code"]).size())
assert (dupes<=1).all(), "Duplicate shred letters exist after canon"

# ============ write ============
full.to_parquet(f"{OUT_BASE}.parquet", index=False)
full.to_csv(f"{OUT_BASE}.csv", index=False, encoding="utf-8")
with open(f"{OUT_BASE}.jsonl","w",encoding="utf-8") as f:
    for _, r in full.iterrows():
        meta = r.to_dict()
        doc_id = meta.pop("id")
        text = meta.pop("text")
        f.write(json.dumps({"id": doc_id, "text": text, "metadata": meta}, ensure_ascii=False) + "\n")

print("Rows:", len(full), " Base:", (full['row_type']=='base').sum(), " Shred:", (full['row_type']=='shred').sum())
print("Wrote:", f"{OUT_BASE}.parquet", f"{OUT_BASE}.csv", f"{OUT_BASE}.jsonl")
