def find_section_spans(lines_after_title):
    """
    Return dict like {'1': (s1_i, s1_end), '2': (s2_i, s2_end), '3': (s3_i, s3_end)}
    where indices are into lines_after_title.
    """
    starts = {}
    for i, ln in enumerate(lines_after_title):
        if '1' not in starts and SECTION_1_RE.search(ln): starts['1'] = i
        if '2' not in starts and SECTION_2_RE.search(ln): starts['2'] = i
        if '3' not in starts and SECTION_3_RE.search(ln): starts['3'] = i

    # Nothing found
    if not starts: return {}

    # Compute ends: next section start or next ALL-CAPS title
    keys = []
    for k,v in starts.items(): keys.append((v,k))
    keys.sort()  # by start index
    ends = {}
    N = len(lines_after_title)
    for idx,(start_i, sk) in enumerate(keys):
        # default end is next section start
        end_i = N
        if idx+1 < len(keys):
            end_i = keys[idx+1][0]
        # but stop earlier if a new ALL-CAPS title appears
        for j in range(start_i+1, end_i):
            if is_all_caps_title(lines_after_title[j]):
                end_i = j
                break
        ends[sk] = (start_i, end_i)
    return ends
# pip install pdfplumber pandas
import re, hashlib
from pathlib import Path
import pdfplumber
import pandas as pd

# --------- CONFIG ----------
PDF_PATH = r"C:\Users\krced\EnlistedClass\afecdpdf\Air Force Enlisted Classification Directory AFECD.pdf"
START_AT_PAGE = 23
END_AT_PAGE   = 368
OUT_CSV = r"C:\Users\krced\EnlistedClass\afecd_to_vectordb\afsc_specialty_text.csv"

EFFECTIVE_DATE = "2025-04-30"
SERVICE = "USAF"

# --------- helpers ----------
# --- normalizer: strips the star symbol and fixes split hyphens ---
def norm_ws(s: str) -> str:
    if not s: return ""
    s = "".join(" " if ch.isspace() else ch for ch in s)
    s = (s.replace("\u200b","").replace("\u200c","").replace("\u200d","")
           .replace("\ufeff","").replace("\u2060","").replace("","")
    )
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(?<=\w)-\s+(?=\w)", "-", s)  # “through- flight” -> “through-flight”
    return s

# SDI codes like "8A200"
SDI_HEAD_RE = re.compile(r"^SDI\s+([0-9A-Z]{2}\d{3})\b", re.I)
SDI_RE      = re.compile(r"\b([0-9A-Z]{2}\d{3})\b", re.I)  # used when scanning lines near title



# Section headers – include "Special Duty Summary" too
SECTION_1_RE = re.compile(r"^1\.\s*[\W_]*(Specialty|Special Duty)\s+Summary", re.I)
SECTION_2_RE = re.compile(r"^2\.\s*[\W_]*Duties and Responsibilities", re.I)
SECTION_3_RE = re.compile(r"^3\.\s*[\W_]*Specialty Qualif", re.I)  # matches Qualification/Qualifications
STOP_AT_3_RE = re.compile(r"^\s*3\.\s", re.I)  # guard so duties don’t swallow section 3

# Flexible section header finder (handles split 1./2./3. lines)
HEADER_NUM_ONLY_RE = re.compile(r"^\s*([123])\.\s*$")

def find_section_header(lines, start_idx, number, label_pattern):
    """
    Finds a section header starting at lines[start_idx].
    Handles both:
      A) '2. Duties and Responsibilities: ...'
      B) '2.' on one line, 'Duties and Responsibilities: ...' on the next
    Returns (pos, header_rest, next_idx) or (None, "", start_idx) if not found.
    """
    i = start_idx
    while i < len(lines):
        ln = lines[i]
        # Case A: combined
        m = re.search(rf"^{number}\.\s*(.*)$", ln, flags=re.I)
        if m and label_pattern.search(ln):
            # strip the label from the same line, keep any trailing remainder
            header_rest = label_pattern.sub("", m.group(1)).strip(" :")
            return i, header_rest, i + 1

        # Case B: split 'N.' then label on next line
        if HEADER_NUM_ONLY_RE.fullmatch(ln) and ln.strip().startswith(f"{number}."):
            if i + 1 < len(lines) and label_pattern.search(lines[i+1]):
                header_rest = label_pattern.sub("", lines[i+1]).strip(" :")
                return i, header_rest, i + 2

        # Bail out if we hit an ALL-CAPS title (a new block) before finding it
        if is_all_caps_title(ln):
            return None, "", start_idx
        i += 1

    return None, "", start_idx

# AFSC/Experience regexes (tolerant versions, in scope for all uses)
AFSC_RE = re.compile(r"\b\d[A-Z]\d(?:\d\d|X\d|00)\*?\b")
EXPERIENCE_HEAD_RE = re.compile(r"^3\.(\d+)\.\s*Experience\.", re.I)
EXPER_ITEM_RE = re.compile(r"^3\.\d+\.(\d+)\.\s*(\d[A-Z]\d(?:\d\d|X\d|00)\*?)\.\s*(.*)$", re.I)


def keep_line(line: str) -> bool:
    t = line.strip()
    if t.startswith("DAFECD,"): return False
    if re.match(r"^\d{1,3}\s+DAFECD,", t, re.I): return False
    if re.match(r"^Page\s+\d+\s*$", t, re.I): return False
    return True

def parse_idx(idx: str):
    parts = re.findall(r"\d+", str(idx))
    return "|".join(f"{int(p):03d}" for p in parts) if parts else "999"



def skill_from_code(code: str) -> str:
    c = str(code).strip().rstrip("*")
    if len(c) < 5:
        return ""
    if c.endswith("00"):
        return "00"
    return c[4] if c[3] == "X" else c[3]

skill_name_map = {"3":"Apprentice/Semiskilled","5":"Journeyman/Skilled","7":"Craftsman/Advanced","9":"Superintendent","00":"Chief Enlisted Manager", "SDI": "Special Duty Identifier"}

def make_id(afsc_code, skill_level_num, row_type, section_index, specialty_title, text):
    base = "|".join([afsc_code or "", skill_level_num or "", row_type or "", section_index or "", specialty_title or "", (text or "")[:120]])
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]

def major_from_afsc(code: str):
    c = code.strip().rstrip("*")
    return (c[:2], c[:3]) if len(c) >= 3 else ("", "")

# fast lines (avoid extract_text)
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

# title detection: accept split titles and ignore AFSC/CEM/DAFECD/parenthetical lines
def is_all_caps_title(line: str) -> bool:
    t = norm_ws(line)
    if len(t) < 3: return False
    if t.upper() != t:  # must be all caps line
        return False
    if t.startswith(("CEM CODE","AFSC ","DAFECD","(")):  # not a title
        return False
    letters = re.sub(r"[^A-Za-z]", "", t)
    return bool(letters)

def collect_until(lines, stop_pred):
    buff, i = [], 0
    while i < len(lines):
        if stop_pred(lines[i]): break
        buff.append(lines[i]); i += 1
    return norm_ws(" ".join(buff)), i

def split_duties(duties_text: str):
    chunks = []
    pat = r"\b2\.\s*(\d+)\.\s*"   # matches 2.1. / 2 . 1 . etc
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
            chunks.append(("2", full))  # single unnumbered duty block
    return chunks


def collect_codes_around_title(lines, title_start_idx, title_end_idx, is_sdi_block=False):
    pre   = lines[:title_start_idx]
    after = lines[title_end_idx:]

    stop = len(after)
    for i, ln in enumerate(after):
        if SECTION_1_RE.search(ln) or SECTION_2_RE.search(ln) or SECTION_3_RE.search(ln):
            stop = i
            break
    post = after[:stop]

    codes = []
    for ln in pre + post:
        codes.extend(AFSC_RE.findall(ln))
        if is_sdi_block:
            m = SDI_HEAD_RE.search(ln)
            if m:
                codes.append(m.group(1))

    seen = set()
    return [c for c in codes if not (c in seen or seen.add(c))]

def parse_block(lines, pstart, pend, source_file):
    rows = []
    # find first ALL CAPS line -> start of title, and join consecutive ALL CAPS lines
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

    # --- skip career-field introduction pages (not tied to a specific AFSC) ---
    if ("CAREER FIELD" in specialty_title
        and title_end_idx < len(lines)
        and re.match(r"^Introduction$", lines[title_end_idx], re.I)):
        return rows  # ignore this block completely

    # detect if this block is SDI
    header_slice = lines[:title_idx]
    m_sdi = next((SDI_HEAD_RE.match(ln) for ln in header_slice if SDI_HEAD_RE.match(ln)), None)
    is_sdi_block = bool(m_sdi)
    sdi_code = m_sdi.group(1) if m_sdi else None

    # Collect AFSCs or SDI around the title
    header_codes = collect_codes_around_title(lines, title_idx, title_end_idx, is_sdi_block)

    # Fan-out logic: SDI block = just the SDI code; else AFSC 3/5/7/9 only
    fanout_codes = []
    if is_sdi_block:
        if sdi_code:
            fanout_codes = [sdi_code]
    else:
        for c in header_codes:
            lvl = skill_from_code(c)
            if lvl in {"3","5","7","9"}:
                fanout_codes.append(c)

    after = lines[title_end_idx:]

    # --- Section 1: Summary ---
    idx = 0
    s1_pos, s1_header_rest, idx_after_1 = find_section_header(after, idx, 1, SECTION_1_RE)
    summary_text = ""
    if s1_pos is not None:
        idx = idx_after_1
        body_text, consumed = collect_until(
            after[idx:], lambda s: SECTION_2_RE.search(s) or SECTION_3_RE.search(s) or is_all_caps_title(s)
        )
        summary_text = norm_ws((s1_header_rest + " " + body_text).strip())
        idx += consumed

        for code in fanout_codes:
            if is_sdi_block:
                lvl = "SDI"
                lvl_name = skill_name_map.get("SDI","")
            else:
                lvl = skill_from_code(code)
                lvl_name = skill_name_map.get(lvl,"")
                if not lvl:
                    continue
            afsc_plain = code.rstrip("*")
            major2, family3 = major_from_afsc(code)
            rows.append({
                "row_type":"summary","section":"1","section_index":"1",
                "section_label":"Specialty Summary","section_sort":parse_idx("1"),
                "specialty_title":specialty_title,
                "service":SERVICE,"major_group_code":major2,"afsc_family":family3,
                "afsc_code":code,"afsc_code_plain":afsc_plain,
                "skill_level_num":lvl,"skill_level_name":lvl_name,
                "text":summary_text,"page_start":str(pstart),"page_end":str(pend),
                "effective_date":EFFECTIVE_DATE,"source_file":source_file,
                "search_key":afsc_plain,"join_key_afsc_skill":f"{afsc_plain}|{lvl}",
                "id":make_id(afsc_plain,lvl,"summary","1",specialty_title,summary_text),
            })

    # --- Section 2: Duties ---
    s2_pos, s2_header_rest, idx_after_2 = find_section_header(after, idx, 2, SECTION_2_RE)
    if s2_pos is not None:
        idx = idx_after_2
        body_text, consumed = collect_until(after[idx:], lambda s: STOP_AT_3_RE.match(s) or is_all_caps_title(s))
        idx += consumed
        duties_text = norm_ws((s2_header_rest + " " + body_text).strip())

        for code in fanout_codes:
            if is_sdi_block:
                lvl = "SDI"
                lvl_name = skill_name_map.get("SDI","")
            else:
                lvl = skill_from_code(code)
                lvl_name = skill_name_map.get(lvl,"")
            afsc_plain = code.rstrip("*")
            major2, family3 = major_from_afsc(code)
            for (lab, txt) in split_duties(duties_text):
                rows.append({
                    "row_type":"duty","section":"2","section_index":lab,
                    "section_label":"Duties and Responsibilities","section_sort":parse_idx(lab),
                    "specialty_title":specialty_title,
                    "service":SERVICE,"major_group_code":major2,"afsc_family":family3,
                    "afsc_code":code,"afsc_code_plain":afsc_plain,
                    "skill_level_num":lvl,"skill_level_name":lvl_name,
                    "text":txt,"page_start":str(pstart),"page_end":str(pend),
                    "effective_date":EFFECTIVE_DATE,"source_file":source_file,
                    "search_key":afsc_plain,"join_key_afsc_skill":f"{afsc_plain}|{lvl}",
                    "id":make_id(afsc_plain,lvl,"duty",lab,specialty_title,txt),
                })

    # --- Section 3: Experience (Specialty Qualifications) ---
    q_pos, _s3_hdr_rest, idx_after_3 = find_section_header(after, idx, 3, SECTION_3_RE)
    if q_pos is not None:
        # capture everything after the header line(s) up to next ALL-CAPS title
        qual_lines = []
        k = idx_after_3
        while k < len(after) and not is_all_caps_title(after[k]):
            qual_lines.append(after[k]); k += 1
        qual_lines = [ln.replace("","") for ln in qual_lines]  # <-- kill the star

        # Find the Experience subheader "3.<N>. Experience."
        exp_start, exp_number = None, None
        for i2, ln in enumerate(qual_lines):
            m = EXPERIENCE_HEAD_RE.match(ln)
            if m:
                exp_start, exp_number = i2 + 1, m.group(1)
                break

        if exp_start is not None:
            sub = qual_lines[exp_start:]

            # Normalize lines so that if a bare 'AFSC.' line appears before '3.x.y.',
            # we fold it into that item. Also handle items that already include the code.
            norm = []
            pending_code = None
            CODE_ONLY_RE = re.compile(rf"^\s*({AFSC_RE.pattern})\.\s*$", re.I)
            ITEM_FLEX_RE = re.compile(rf"^3\.\d+\.(\d+)\.\s*(?:({AFSC_RE.pattern})\.\s*)?(.*)$", re.I)

            i = 0
            while i < len(sub):
                ln = sub[i]
                m_code = CODE_ONLY_RE.match(ln)
                m_item = ITEM_FLEX_RE.match(ln)
                if m_code:
                    pending_code = m_code.group(1)
                    i += 1
                    continue
                if m_item:
                    item_subnum = m_item.group(1)
                    code_here = m_item.group(2) or pending_code
                    rest = m_item.group(3).strip()
                    pending_code = None
                    if code_here:
                        # Build a synthetic standard line: "3.x.y. CODE. rest"
                        norm.append((item_subnum, code_here, rest))
                    else:
                        # No code? skip this item
                        i += 1
                        continue
                else:
                    # continuation text for last norm item, if any
                    if norm:
                        last = norm[-1]
                        norm[-1] = (last[0], last[1], norm_ws((last[2] + " " + ln).strip()))
                    i += 1
                    continue
                i += 1

            # Emit rows
            for subnum, afsc_in_item, tail in norm:
                lvl = skill_from_code(afsc_in_item)
                if not lvl or lvl == "1":
                    continue
                afsc_plain = afsc_in_item.rstrip("*")
                major2, family3 = major_from_afsc(afsc_in_item)
                section_index = f"3.{exp_number}.{subnum}"
                exp_text = norm_ws(tail)
                rows.append({
                    "row_type": "experience",
                    "section": "3",
                    "section_index": section_index,
                    "section_label": "Experience",
                    "section_sort": parse_idx(section_index),
                    "specialty_title": specialty_title,
                    "afsc_code": afsc_in_item,
                    "afsc_code_plain": afsc_plain,
                    "skill_level_num": lvl,
                    "skill_level_name": skill_name_map.get(lvl, ""),
                    "service": SERVICE,
                    "major_group_code": major2,
                    "afsc_family": family3,
                    "text": exp_text,
                    "page_start": str(pstart),
                    "page_end": str(pend),
                    "effective_date": EFFECTIVE_DATE,
                    "source_file": source_file,
                    "search_key": afsc_plain,
                    "join_key_afsc_skill": f"{afsc_plain}|{lvl}",
                    "id": make_id(afsc_plain, lvl, "experience", section_index, specialty_title, exp_text),
                })
    # Section 3 handled above with find_section_spans and '3' in spans

    return rows

# --------- drive the PDF ----------
all_rows = []
with pdfplumber.open(PDF_PATH) as pdf:
    source_file = Path(PDF_PATH).name
    n = len(pdf.pages)
    if END_AT_PAGE < START_AT_PAGE:
        raise ValueError("END_AT_PAGE must be >= START_AT_PAGE")

    pages = []
    end_exclusive = min(n, END_AT_PAGE)
    for pi in range(START_AT_PAGE - 1, end_exclusive):
        p = pdf.pages[pi]
        lines = words_by_lines(p, x_tol=2, y_tol=3)
        pages.append((pi + 1, lines))

    def is_header_line(s: str) -> bool:
        t = s.strip()
        return t.startswith("AFSC ") or t.startswith("CEM Code") or t.upper().startswith("SDI ")

    cur_lines = []
    block_pstart = None
    got_title = False    # flips true once we’ve seen the specialty title in this block

    def flush_block(pstart, pend, lines):
        if not lines:
            return
        all_rows.extend(parse_block(lines, pstart, pend, source_file))

    last_pnum = None
    for (pnum, lines) in pages:
        last_pnum = pnum
        for ln in lines:
            if is_header_line(ln):
                # new header starts a new block only if we already had a title (i.e., we’re leaving a finished block)
                if cur_lines and got_title:
                    flush_block(block_pstart or pnum, pnum, cur_lines)
                    cur_lines = []
                    got_title = False
                    block_pstart = pnum
                elif not cur_lines:
                    block_pstart = pnum
                cur_lines.append(ln)
                continue

            # track title appearance, but do NOT split on titles anymore
            if is_all_caps_title(ln):
                got_title = True

            if not cur_lines:
                # occasionally a page may begin mid-block without a header line; start capturing anyway
                block_pstart = block_pstart or pnum
            cur_lines.append(ln)

    # final flush within the bound
    if cur_lines and last_pnum is not None:
        flush_block(block_pstart or pages[0][0], last_pnum, cur_lines)

print(f"Parsed pages {START_AT_PAGE}..{min(n, END_AT_PAGE)}")

# --------- build + write ----------
df = pd.DataFrame(all_rows)
if df.empty:
    print("No rows extracted. Check parsing logic or page range.")
else:
    # drop dup ids and empty text
    df = df.drop_duplicates(subset=["id"])
    df = df[df["text"].astype(str).str.strip() != ""]

    # --- Enrich with AFSC catalog for major_group_name and afsc_title ---
    try:
        base = pd.read_parquet(r"C:\Users\krced\EnlistedClass\afsc_schred_to_vectordb\afsc_catalog_enriched.parquet")
        base = base[base["row_type"]=="base"][["afsc_code_plain","skill_level_num","major_group_name","afsc_title"]].drop_duplicates()
        df = df.merge(base, on=["afsc_code_plain","skill_level_num"], how="left")
    except Exception as e:
        print("Note: couldn’t enrich with AFSC titles:", e)

    # --- Column order for correct output ---
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

    df = df[order_cols].sort_values(
        ["afsc_code_plain","skill_level_num","row_type","section_sort"]
    )
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    # --- Tiny QA ---
    print("Row counts by type:\n", df["row_type"].value_counts())
    bad = (df.pivot_table(index=["afsc_code_plain","skill_level_num"],
                         columns="row_type", values="id", aggfunc="count", fill_value=0)
             .query("summary == 0 or duty == 0"))
    if not bad.empty:
        print("Heads-up: some AFSCs missing summary/duty:\n", bad.head(10))

    # --- Minimal QA before writing ---
    cem_leaks = df.query("skill_level_num == '00' and row_type in ['duty','summary']")
    assert cem_leaks.empty, f"CEM leaked:\n{cem_leaks[['afsc_code_plain','row_type']].drop_duplicates()}"

    # --- Minimal QA for SDI pages (309–368) ---
    sdi_rows = df[df["skill_level_num"]=="SDI"]
    print("SDI rows:", len(sdi_rows))
    print(sdi_rows["row_type"].value_counts())
    # example: 8A200 ENLISTED AIDE should have summary + duties (experience likely 'Not used')
    print(sdi_rows[sdi_rows["afsc_code_plain"]=="8A200"][ ["row_type","section_index","specialty_title","text"] ].head(8).to_string(index=False))

    # --- Quick re-check targets for problem ladders ---
    check = df.pivot_table(index=["afsc_code_plain","skill_level_num"],
                          columns="row_type", values="id",
                          aggfunc="count", fill_value=0)

    print(check.loc[["1C131","1C151","1C171","1C191"]].fillna(0))
    print(check.loc[["1C731","1C751","1C771","1C791"]].fillna(0))
    print(check.loc[["1D735","1D755"]].fillna(0))

    # --- Quick sanity to prove 1C7xx is fixed ---
    blk = df.pivot_table(index=["afsc_code_plain","skill_level_num"],
                        columns="row_type", values="id", aggfunc="count", fill_value=0)
    print(blk.loc[["1C731","1C751","1C771","1C791"]].fillna(0))

    # --- SDI verification ---
    sdi_df = df[df["skill_level_num"]=="SDI"]
    print("SDI rows:", len(sdi_df))
    if not sdi_df.empty:
        print(sdi_df["row_type"].value_counts())
        print(sdi_df.head(10)[["afsc_code_plain","row_type","section","section_index","specialty_title"]].to_string(index=False))
