# sdi_extract.py
# Standalone SDI (8X***) parser for AFECD -> vector-ready CSV

import re
import hashlib
from pathlib import Path
import pdfplumber
import pandas as pd

print("[RUNNING]", __file__)

# ---------- CONFIG ----------
PDF_PATH = r"C:\Users\krced\EnlistedClass\afecdpdf\Air Force Enlisted Classification Directory AFECD.pdf"
SDI_START_PAGE = 309       # inclusive (1-based)
SDI_END_PAGE   = 368       # inclusive (1-based)
OUT_CSV = r"C:\Users\krced\EnlistedClass\afecd_to_vectordb\sdi_specialty_text.csv"

EFFECTIVE_DATE = "2025-04-30"
SERVICE = "USAF"
SKILL_FOR_SDI = ""         # set to "" for clean vectordb; or "SDI" if you want it labeled

# ---------- REGEX ----------
AFSC_RE          = re.compile(r"\b\d[A-Z]\d(?:\d\d|X\d|00)\*?\b")
SDI_CODE_RE      = re.compile(r"^8[A-Z]\d{3}$", re.I)
SDI_INLINE_RE    = re.compile(r"\b8[A-Z]\d{3}\*?\b", re.I)

SECTION_1_RE     = re.compile(r"\b1\.\s*(?:Specialty|Special\s*Duty)\s+Summary", re.I)
SECTION_2_RE     = re.compile(r"\b2\.\s*Duties\s*and\s*Responsibilities", re.I)
SECTION_3_RE     = re.compile(r"\b3\.\s*Specialty\s+Qualif", re.I)  # Qualif(ication|ications)
DUTIES_FLEX_RE   = re.compile(r"(?:Duties\s*(?:\b2\.\b\s*)?and\s*Responsibilities|Career\s*Field\s*Core\s*Functions)", re.I)
HEADER_NUM_ONLY  = re.compile(r"^\s*([123])\.\s*$", re.I)
STOP_AT_3_RE     = re.compile(r"^\s*3\.\s", re.I)
EXPER_HEAD_RE    = re.compile(r"^3\.(\d+)\.\s*Experience\.", re.I)

# ---------- HELPERS ----------
def norm_ws(s: str) -> str:
    if not s: return ""
    s = "".join(" " if ch.isspace() else ch for ch in s)
    s = (s.replace("\u200b","").replace("\u200c","").replace("\u200d","")
           .replace("\ufeff","").replace("\u2060","").replace("",""))
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(?<=\w)-\s+(?=\w)", "-", s)  # through- flight -> through-flight
    return s

def keep_line(line: str) -> bool:
    t = line.strip()
    if t.startswith("DAFECD,"): return False
    if re.match(r"^\d{1,3}\s+DAFECD,", t, re.I): return False
    if re.match(r"^Page\s+\d+\s*$", t, re.I): return False
    return True

def words_by_lines(page, x_tol=2, y_tol=3):
    words = page.extract_words(use_text_flow=True, keep_blank_chars=False,
                               x_tolerance=x_tol, y_tolerance=y_tol)
    rows = {}
    for w in words:
        y = round(w["top"], 0)
        rows.setdefault(y, []).append(w)
    out = []
    for y in sorted(rows):
        ws = sorted(rows[y], key=lambda w: w["x0"])
        line = norm_ws(" ".join(w["text"] for w in ws))
        if line and keep_line(line):
            out.append(line)
    return out

def is_all_caps_title(line: str) -> bool:
    t = norm_ws(line)
    if len(t) < 3: return False
    if t.upper() != t: return False
    if t.startswith(("CEM CODE","AFSC ","DAFECD","(")): return False
    letters = re.sub(r"[^A-Za-z]", "", t)
    return bool(letters)

def parse_idx(idx: str):
    parts = re.findall(r"\d+", str(idx))
    return "|".join(f"{int(p):03d}" for p in parts) if parts else "999"

def make_id(code, lvl, row_type, section_index, title, text):
    base = "|".join([code or "", lvl or "", row_type or "", section_index or "", title or "", (text or "")[:120]])
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]

def is_sdi_code(code: str) -> bool:
    c = str(code or "").replace(" ", "").strip().rstrip("*").upper()
    return bool(SDI_CODE_RE.fullmatch(c))

def pick_primary_sdi(codes):
    """
    Prefer 8X100/8X200/... over 8X000; both over 8X9x.
    """
    if not codes: return None
    uniq = []
    seen = set()
    for c in codes:
        c = c.upper().rstrip("*")
        if c not in seen:
            seen.add(c); uniq.append(c)

    def score(c):
        if not c.endswith("00"): return 0
        # e.g., 8G100 => '1', 8G000 => '0'
        return 2 if c[2] != "0" else 1

    uniq.sort(key=lambda x: (-score(x), codes.index(x)))
    return uniq[0]

def collect_until(lines, stop_pred):
    buff, i = [], 0
    while i < len(lines):
        if stop_pred(lines[i]): break
        buff.append(lines[i]); i += 1
    return norm_ws(" ".join(buff)), i

def split_duties(duties_text: str):
    chunks = []
    pat = r"\b2\.\s*(\d+)\.\s*"
    for m in re.finditer(pat, duties_text):
        idx = f"2.{m.group(1)}"
        start = m.end()
        next_m = re.search(pat, duties_text[start:])
        end = start + next_m.start() if next_m else len(duties_text)
        txt = norm_ws(duties_text[start:end])
        if txt:
            chunks.append((idx, txt))
    if not chunks:
        full = norm_ws(duties_text)
        if full:
            chunks.append(("2", full))
    return chunks

def find_section_header(lines, start_idx, number, label_pattern):
    """
    A) 'N. <label> ...'
    B) 'N.' on one line, '<label> ...' next
    C) 'Duties' / '2.' / 'and Responsibilities' split across lines
    """
    i = start_idx
    while i < len(lines):
        ln = lines[i]

        # A: combined line
        m = re.search(rf"^\s*{number}\.\s*(.*)$", ln, flags=re.I)
        if m and (label_pattern.search(ln) or (number == 2 and DUTIES_FLEX_RE.search(ln))):
            header_rest = label_pattern.sub("", m.group(1)).strip(" :")
            return i, header_rest, i + 1

        # B: 'N.' alone then label
        if HEADER_NUM_ONLY.fullmatch(ln) and ln.strip().startswith(f"{number}."):
            if i + 1 < len(lines) and (label_pattern.search(lines[i+1]) or (number == 2 and DUTIES_FLEX_RE.search(lines[i+1]))):
                header_rest = label_pattern.sub("", lines[i+1]).strip(" :")
                return i, header_rest, i + 2

        # C: windowed flex for weird splits (esp. Section 2)
        prev = lines[i-1] if i > 0 else ""
        nxt  = lines[i+1] if i + 1 < len(lines) else ""
        window = norm_ws(f"{prev} {ln} {nxt}")
        has_num = re.search(rf"\b{number}\.\b", window)
        has_lbl = (DUTIES_FLEX_RE.search(window) if number == 2 else label_pattern.search(window))
        if has_num and has_lbl:
            # align to the 'N.' line if possible
            if re.match(rf"^\s*{number}\.\s*", ln):  # current is num line
                start_pos = i - 1 if (i > 0 and re.search(r"Duties$", prev, re.I)) else i
                return start_pos, "", i + 1
            if i > 0 and re.match(rf"^\s*{number}\.\s*$", prev):
                return i - 1, "", i
            if i + 1 < len(lines) and re.match(rf"^\s*{number}\.\s*", nxt):
                return i, "", i + 2

        if is_all_caps_title(ln):
            return None, "", start_idx
        i += 1

    return None, "", start_idx

# ---------- SDI PARSER ----------
def collect_codes_around_title(lines, title_start_idx, title_end_idx):
    pre   = lines[:title_start_idx]
    after = lines[title_end_idx:]

    # stop after first section header or next ALL-CAPS title
    stop = len(after)
    for i, ln in enumerate(after):
        if SECTION_1_RE.search(ln) or SECTION_2_RE.search(ln) or SECTION_3_RE.search(ln) or is_all_caps_title(ln):
            stop = i
            break
    post = after[:stop]

    cand = []
    for ln in (pre + post):
        cand.extend(m.group(0) for m in SDI_INLINE_RE.finditer(ln))
    return cand

def parse_sdi_block(lines, pstart, pend, source_file):
    rows = []

    # Find Title
    title_idx = None
    for i, ln in enumerate(lines):
        if is_all_caps_title(ln):
            title_idx = i
            break
    if title_idx is None:
        return rows

    title_lines = []
    j = title_idx
    while j < len(lines) and is_all_caps_title(lines[j]):
        title_lines.append(norm_ws(lines[j])); j += 1
    specialty_title = norm_ws(" ".join(title_lines))
    title_end_idx = j

    # Find SDI code (title + header + vicinity)
    header_slice = lines[:title_idx]
    sdi_near = collect_codes_around_title(lines, title_idx, title_end_idx)

    # include any codes shown on header lines like "SDI 8G000 8G091 ..."
    for ln in header_slice:
        for m in SDI_INLINE_RE.finditer(ln):
            sdi_near.append(m.group(0))

    # also look inside the title itself (some are like "SDI 8G100 BASE HONOR GUARD PROGRAM MANAGER")
    for m in SDI_INLINE_RE.finditer(" ".join(title_lines)):
        sdi_near.append(m.group(0))

    sdi_codes = [c.upper().rstrip("*") for c in sdi_near if is_sdi_code(c)]
    sdi_code = pick_primary_sdi(sdi_codes)
    if not sdi_code:
        # No SDI code discovered -> skip quietly
        # print(f"[WARN] No SDI code found for title '{specialty_title}' p{pstart}-{pend}")
        return rows

    # Sections (from after title)
    after = lines[title_end_idx:]

    # Section 1: Summary
    idx = 0
    s1_pos, s1_header_rest, idx_after_1 = find_section_header(after, idx, 1, SECTION_1_RE)
    summary_text = ""
    if s1_pos is not None:
        idx = idx_after_1
        body, consumed = collect_until(
            after[idx:], lambda s: SECTION_2_RE.search(s) or SECTION_3_RE.search(s) or is_all_caps_title(s)
        )
        summary_text = norm_ws((s1_header_rest + " " + body).strip())
        idx += consumed

        rows.append({
            "row_type":"summary","section":"1","section_index":"1",
            "section_label":"Specialty Summary","section_sort":parse_idx("1"),
            "specialty_title":specialty_title,
            "service":SERVICE,"major_group_code":"","major_group_name":"",
            "afsc_family":"","afsc_code":sdi_code,"afsc_code_plain":sdi_code,
            "skill_level_num":SKILL_FOR_SDI,"skill_level_name":("Special Duty Identifier" if SKILL_FOR_SDI else ""),
            "text":summary_text,"page_start":str(pstart),"page_end":str(pend),
            "effective_date":EFFECTIVE_DATE,"source_file":source_file,
            "search_key":sdi_code,"join_key_afsc_skill":f"{sdi_code}|{SKILL_FOR_SDI}",
            "id":make_id(sdi_code,SKILL_FOR_SDI,"summary","1",specialty_title,summary_text),
        })

    # Section 2: Duties
    s2_pos, s2_header_rest, idx_after_2 = find_section_header(after, idx, 2, SECTION_2_RE)
    if s2_pos is not None:
        idx = idx_after_2
        body, consumed = collect_until(after[idx:], lambda s: STOP_AT_3_RE.match(s) or is_all_caps_title(s))
        idx += consumed
        duties_text = norm_ws((s2_header_rest + " " + body).strip())

        for (lab, txt) in split_duties(duties_text):
            rows.append({
                "row_type":"duty","section":"2","section_index":lab,
                "section_label":"Duties and Responsibilities","section_sort":parse_idx(lab),
                "specialty_title":specialty_title,
                "service":SERVICE,"major_group_code":"","major_group_name":"",
                "afsc_family":"","afsc_code":sdi_code,"afsc_code_plain":sdi_code,
                "skill_level_num":SKILL_FOR_SDI,"skill_level_name":("Special Duty Identifier" if SKILL_FOR_SDI else ""),
                "text":txt,"page_start":str(pstart),"page_end":str(pend),
                "effective_date":EFFECTIVE_DATE,"source_file":source_file,
                "search_key":sdi_code,"join_key_afsc_skill":f"{sdi_code}|{SKILL_FOR_SDI}",
                "id":make_id(sdi_code,SKILL_FOR_SDI,"duty",lab,specialty_title,txt),
            })

    # Section 3: Experience (rare for SDI, but we’ll grab if present)
    q_pos, _s3_rest, idx_after_3 = find_section_header(after, idx, 3, SECTION_3_RE)
    if q_pos is not None:
        qual_lines = []
        k = idx_after_3
        while k < len(after) and not is_all_caps_title(after[k]):
            qual_lines.append(after[k]); k += 1
        exp_start, exp_number = None, None
        for i2, ln in enumerate(qual_lines):
            m = EXPER_HEAD_RE.match(ln)
            if m:
                exp_start, exp_number = i2 + 1, m.group(1)
                break
        if exp_start is not None:
            exp_text = norm_ws(" ".join(qual_lines[exp_start:]))
            if exp_text:
                section_index = f"3.{exp_number}"
                rows.append({
                    "row_type":"experience","section":"3","section_index":section_index,
                    "section_label":"Experience","section_sort":parse_idx(section_index),
                    "specialty_title":specialty_title,
                    "service":SERVICE,"major_group_code":"","major_group_name":"",
                    "afsc_family":"","afsc_code":sdi_code,"afsc_code_plain":sdi_code,
                    "skill_level_num":SKILL_FOR_SDI,"skill_level_name":("Special Duty Identifier" if SKILL_FOR_SDI else ""),
                    "text":exp_text,"page_start":str(pstart),"page_end":str(pend),
                    "effective_date":EFFECTIVE_DATE,"source_file":source_file,
                    "search_key":sdi_code,"join_key_afsc_skill":f"{sdi_code}|{SKILL_FOR_SDI}",
                    "id":make_id(sdi_code,SKILL_FOR_SDI,"experience",section_index,specialty_title,exp_text),
                })

    return rows

# ---------- SCAN PDF ----------
all_rows = []
with pdfplumber.open(PDF_PATH) as pdf:
    source_file = Path(PDF_PATH).name
    n = len(pdf.pages)

    if SDI_END_PAGE < SDI_START_PAGE:
        raise ValueError("SDI_END_PAGE must be >= SDI_START_PAGE")

    start_i = max(1, SDI_START_PAGE) - 1
    end_i   = min(n, SDI_END_PAGE)   # inclusive (1-based)
    pages = []
    for pi in range(start_i, end_i):
        p = pdf.pages[pi]
        lines = words_by_lines(p, x_tol=2, y_tol=3)
        pages.append((pi + 1, lines))

    def is_header_line(s: str) -> bool:
        t = s.strip()
        return t.upper().startswith("SDI ")

    cur_lines = []
    block_pstart = None
    got_title = False

    def flush_block(pstart, pend, lines):
        if not lines: return
        rows = parse_sdi_block(lines, pstart, pend, source_file)
        all_rows.extend(rows)

    last_pnum = None
    for (pnum, lines) in pages:
        last_pnum = pnum
        for ln in lines:
            if is_header_line(ln):
                if cur_lines and got_title:
                    flush_block(block_pstart or pnum, pnum, cur_lines)
                    cur_lines = []
                    got_title = False
                    block_pstart = pnum
                elif not cur_lines:
                    block_pstart = pnum
                cur_lines.append(ln)
                continue

            if is_all_caps_title(ln):
                got_title = True

            if not cur_lines:
                block_pstart = block_pstart or pnum
            cur_lines.append(ln)

    if cur_lines and last_pnum is not None:
        flush_block(block_pstart or pages[0][0], last_pnum, cur_lines)

print(f"Parsed SDI pages {SDI_START_PAGE}..{SDI_END_PAGE}")

# ---------- BUILD + WRITE ----------
df = pd.DataFrame(all_rows)
if df.empty:
    print("No SDI rows extracted. Adjust page range or parsing.")
else:
    # de-dup + no empty text
    df = df.drop_duplicates(subset=["id"])
    df = df[df["text"].astype(str).str.strip() != ""]

    # final column order (matches your AFSC output schema)
    order_cols = [
        "row_type","section","section_index","section_label","section_sort",
        "specialty_title","service","major_group_code","major_group_name",
        "afsc_family","afsc_code","afsc_code_plain",
        "skill_level_num","skill_level_name",
        "text",
        "page_start","page_end","effective_date","source_file",
        "search_key","join_key_afsc_skill","id"
    ]
    for c in order_cols:
        if c not in df.columns: df[c] = ""
    df = df[order_cols].sort_values(["afsc_code_plain","row_type","section_sort"])

    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print("Row counts by type:\n", df["row_type"].value_counts())
    # quick peek
    print(df.head(8)[["afsc_code_plain","row_type","section","section_index","specialty_title"]].to_string(index=False))
