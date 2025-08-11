# --------- IMPORTS ----------
import re
import csv
import sys
import hashlib
from pathlib import Path
import pdfplumber
import pandas as pd

# afecd_wow.py
print("[RUNNING]", __file__)

# ↓ only after imports define regex/constants
DUTY_ANCHOR_RE = re.compile(r"^\s*2\.(\d+)\.\s*(.*)")
IS_SDI_CODE_RE = re.compile(r"^8[0-9A-Z]\d{3}$", re.I)

def is_sdi_code(code: str) -> bool:
    c = str(code or "").replace(" ", "").strip().rstrip("*").upper()
    return bool(IS_SDI_CODE_RE.fullmatch(c))

def skill_from_code(code: str) -> str:
    c = str(code or "").replace(" ", "").strip().rstrip("*").upper()
    if is_sdi_code(c):
        return ""
    if len(c) < 5:
        return ""
    if c.endswith("00"):
        return "00"
    return c[4] if c[3] == "X" else c[3]

def major_from_afsc(code: str):
    c = str(code or "").replace(" ", "").strip().rstrip("*").upper()
    if is_sdi_code(c):
        return ("", "")
    return (c[:2], c[:3]) if len(c) >= 3 else ("", "")

# Tolerant duty extractor
def extract_duties(lines, start_idx, end_idx=None):
    end_idx = end_idx if end_idx is not None else len(lines)

    # Find all "2.x." anchors in the scan range
    anchors = [k for k in range(start_idx, end_idx) if DUTY_ANCHOR_RE.match(lines[k])]

    # Fallback: if none found (because start_idx landed on a stray word like "Performs"),
    # scan a slightly wider window forward/backward to catch the first 2.x.
    if not anchors:
        for k in range(max(0, start_idx - 3), min(end_idx, start_idx + 20)):
            if DUTY_ANCHOR_RE.match(lines[k]):
                anchors.append(k)
                break

    rows = []
    for idx, a in enumerate(anchors):
        a2 = anchors[idx+1] if idx+1 < len(anchors) else end_idx
        m = DUTY_ANCHOR_RE.match(lines[a])
        sec_idx = f"2.{m.group(1)}"
        heading = m.group(2).strip(" :")
        text = "\n".join(lines[a:a2]).strip()
        rows.append((sec_idx, heading, text))
    return rows

# --------- IMPORTS ----------
import re
import csv
import sys
import hashlib
from pathlib import Path
import pdfplumber
import pandas as pd


# Section labels
SECTION_1_RE  = re.compile(r"\b1\.\s*(?:Specialty|Special\s*Duty)?\s*Summary", re.I)
SECTION_2_RE  = re.compile(r"\b2\.\s*(?:Duties\s*and\s*Responsibilities|Career\s*Field\s*Core\s*Functions)", re.I)
SECTION_3_RE  = re.compile(r"\b3\.\s*Specialty\s+Qualif", re.I)

# "Label-only" forms used by the flexible finder
LABEL_1_ONLY_RE = re.compile(r"(?:Specialty|Special\s*Duty)\s*Summary", re.I)
LABEL_2_ONLY_RE = re.compile(r"(?:Duties\s*and\s*Responsibilities|Career\s*Field\s*Core\s*Functions)", re.I)
LABEL_3_ONLY_RE = re.compile(r"Specialty\s+Qualif", re.I)

# Flexible variant for Section 2 that tolerates the 'Duties' / '2.' / 'and Responsibilities' split
DUTIES_FLEX_RE  = re.compile(
    r"(?:Duties\s*(?:\b2\.\b\s*)?and\s*Responsibilities|Career\s*Field\s*Core\s*Functions)",
    re.I
)

HEADER_NUM_ONLY_RE = re.compile(r"^\s*([123])\.\s*$", re.I)
# SDI codes are 8X*** (e.g., 8A200, 8P100, 8U000). Treat ONLY those as SDI.
IS_SDI_CODE_RE = re.compile(r"^8[0-9A-Z]\d{3}$", re.I)

def is_sdi_code(code: str) -> bool:
    c = str(code or "").replace(" ", "").strip().rstrip("*").upper()
    return bool(IS_SDI_CODE_RE.fullmatch(c))
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


# SDI code headers like "SDI 8P100" or split "SDI 8P 100"
SDI_HEAD_RE_COMBINED = re.compile(r"^SDI\s+([0-9A-Z]{2}\d{3})\b", re.I)
SDI_HEAD_RE_SPLIT    = re.compile(r"^SDI\s+([0-9A-Z]{2})\s*(\d)\s*(\d)\s*(\d)\b", re.I)
SDI_INLINE_RE        = re.compile(r"\b([0-9A-Z]{2})\s*(\d)\s*(\d)\s*(\d)\b", re.I)

def _norm_sdi(groups):
    code = f"{groups[0]}{groups[1]}{groups[2]}{groups[3]}".upper()
    return code if re.fullmatch(r"[0-9A-Z]{2}\d{3}", code) else None

def _extract_sdi_codes(lines):
    out, seen = [], set()
    for ln in lines:
        m = SDI_HEAD_RE_COMBINED.match(ln)
        if m:
            c = m.group(1).upper()
            if c not in seen: seen.add(c); out.append(c)
        m2 = SDI_HEAD_RE_SPLIT.match(ln)
        if m2:
            c = _norm_sdi(m2.groups())
            if c and c not in seen: seen.add(c); out.append(c)
        for g in SDI_INLINE_RE.findall(ln):
            c = _norm_sdi(g)
            if c and c not in seen: seen.add(c); out.append(c)
    return out

def _prefer_non_000(sdi_codes):
    if not sdi_codes: return None
    specific = [c for c in sdi_codes if not c.endswith("000")]
    return (specific or sdi_codes)[0]

def collect_codes_around_title(lines, title_start_idx, title_end_idx, is_sdi_block=False):
    """
    AFSC: collect AFSCs near title.
    SDI: pick best SDI near title (prefer non-000).
    """

    title_lines = lines[title_start_idx:title_end_idx]
    pre   = lines[:title_start_idx]
    after = lines[title_end_idx:]

    # stop scanning after first numbered section or next ALL-CAPS title
    stop = len(after)
    for i, ln in enumerate(after):
        if SECTION_1_RE.search(ln) or SECTION_2_RE.search(ln) or SECTION_3_RE.search(ln) or is_all_caps_title(ln):
            stop = i
            break
    post = after[:stop]

    if is_sdi_block:
        # look in pre + title + post for split or combined SDI codes
        sdi_codes = _extract_sdi_codes(pre + title_lines + post)
        best = _prefer_non_000(sdi_codes)
        return [best] if best else []

    # AFSCs
    codes, seen = [], set()
    for ln in (pre + post):
        for c in AFSC_RE.findall(ln):
            if c not in seen:
                seen.add(c); codes.append(c)
    return codes

SECTION_1_RE = re.compile(r"^1\.\s*[\W_]*(Specialty|Special Duty)\s+Summary", re.I)
SECTION_2_RE = re.compile(r"^2\.\s*[\W_]*Duties and Responsibilities", re.I)
SECTION_3_RE = re.compile(r"^3\.\s*[\W_]*Specialty Qualif", re.I)  # matches Qualification/Qualifications
STOP_AT_3_RE = re.compile(r"^\s*3\.\s", re.I)  # guard so duties don’t swallow section 3

# NEW: label-only patterns for split headers (no leading "1./2./3.")


# Label-only patterns for split headers (no leading "1./2./3.")
LABEL_1_ONLY_RE = re.compile(r"(Specialty|Special Duty)\s+Summary", re.I)
LABEL_2_ONLY_RE = re.compile(r"Duties\s*and\s*Responsibilities", re.I)
LABEL_3_ONLY_RE = re.compile(r"Specialty\s+Qualif", re.I)

# Flexible match that tolerates a '2.' wedged between 'Duties' and 'and Responsibilities'
DUTIES_FLEX_RE = re.compile(r"Duties\s*(?:\b2\.\b\s*)?and\s*Responsibilities", re.I)

HEADER_NUM_ONLY_RE = re.compile(r"^\s*([123])\.\s*$")


# Flexible section header finder (handles split 1./2./3. lines and label fragments)
def find_section_header(lines, start_idx, number, label_pattern, label_only_pattern=None):
    """
    Handles:
      A) 'N. <label> ...'
      B) 'N.' on one line, '<label>' on next
      C) Split weirdness like 'Duties' / '2.' / 'and Responsibilities:' or '2. Career Field Core Functions:'
    Returns (pos, header_rest, next_idx) or (None, "", start_idx).
    """
    def norm_ws(s: str) -> str:
        return re.sub(r"\s+", " ", s or "").strip()

    i = start_idx
    while i < len(lines):
        ln = lines[i]

        # A) Combined on one line
        m = re.search(rf"^\s*{number}\.\s*(.*)$", ln, flags=re.I)
        if m and label_pattern.search(ln):
            header_rest = label_pattern.sub("", m.group(1)).strip(" :")
            return i, header_rest, i + 1

        # B) 'N.' then label on next line
        if HEADER_NUM_ONLY_RE.fullmatch(ln) and ln.strip().startswith(f"{number}."):
            if i + 1 < len(lines) and label_pattern.search(lines[i+1]):
                header_rest = label_pattern.sub("", lines[i+1]).strip(" :")
                return i, header_rest, i + 2

        # C) Ultra-flex (3-line window)
        prev = lines[i-1] if i > 0 else ""
        nxt  = lines[i+1] if i + 1 < len(lines) else ""
        window = norm_ws(f"{prev} {ln} {nxt}")

        has_num = re.search(rf"\b{number}\.\b", window)

        if number == 2:
            has_lbl = bool(DUTIES_FLEX_RE.search(window))
        elif number == 1:
            # Accept 'Specialty' on one line and '1. Summary.' on the next
            has_lbl = bool(
                re.search(r"\bSummary\b", window, re.I) and has_num
                or (re.search(r"Specialty", prev, re.I) and has_num)
            )
        else:
            has_lbl = bool((label_only_pattern or label_pattern).search(window))

        if has_num and has_lbl:
            # try to align to the 'N.' line or the 'Duties' line just before it
            if re.match(rf"^\s*{number}\.\s*", ln):
                start_pos = i - 1 if (i > 0 and re.search(r"Duties$", prev, re.I)) else i
                return start_pos, "", i + 1
            if i > 0 and re.match(rf"^\s*{number}\.\s*$", prev):
                return i - 1, "", i
            if i + 1 < len(lines) and re.match(rf"^\s*{number}\.\s*", nxt):
                return i, "", i + 2

        # bail if we hit the next all-caps title
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
    """
    AFSC -> returns 3/5/7/9/00 (never 1)
    SDI  -> '' (no level)
    """
    c = str(code or "").replace(" ", "").strip().rstrip("*").upper()
    if is_sdi_code(c):          # ONLY 8X*** are SDIs
        return ""
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
    """
    AFSC -> (major_group_code, afsc_family)
    SDI  -> ('','')
    """
    c = str(code or "").replace(" ", "").strip().rstrip("*").upper()
    if is_sdi_code(c):
        return ("", "")
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
    """
    AFSC: collect AFSCs near title.
    SDI: pick best SDI near title (prefer non-000).
    """
    title_lines = lines[title_start_idx:title_end_idx]  # <-- must exist
    pre   = lines[:title_start_idx]
    after = lines[title_end_idx:]

    stop = len(after)
    for i, ln in enumerate(after):
        if SECTION_1_RE.search(ln) or SECTION_2_RE.search(ln) or SECTION_3_RE.search(ln) or is_all_caps_title(ln):
            stop = i
            break
def find_section_header(lines, start_idx, number, label_pattern, label_only_pattern=None):
    """
    Handles:
      A) 'N. <label> ...'
      B) 'N.' on one line, '<label>' on next
      C) Weird splits like 'Duties' / '2.' / 'and Responsibilities:'
    Returns (pos, header_rest, next_idx) or (None, "", start_idx).
    """
    i = start_idx
    while i < len(lines):
        ln = lines[i]

        # A) Combined on one line
        m = re.search(rf"^{number}\.\s*(.*)$", ln, flags=re.I)
        if m and label_pattern.search(ln):
            header_rest = label_pattern.sub("", m.group(1)).strip(" :")
            return i, header_rest, i + 1

        # B) 'N.' then label on next line
        if HEADER_NUM_ONLY_RE.fullmatch(ln) and ln.strip().startswith(f"{number}."):
            if i + 1 < len(lines) and label_pattern.search(lines[i+1]):
                header_rest = label_pattern.sub("", lines[i+1]).strip(" :")
                return i, header_rest, i + 2

        # C) Ultra-flex (3-line window)
        prev = lines[i-1] if i > 0 else ""
        nxt  = lines[i+1] if i + 1 < len(lines) else ""
        window = norm_ws(f"{prev} {ln} {nxt}")

        has_num = re.search(rf"\b{number}\.\b", window)
        if number == 2:
            has_lbl = bool(DUTIES_FLEX_RE.search(window))
        else:
            if label_only_pattern is not None:
                has_lbl = bool(label_only_pattern.search(window))
            else:
                has_lbl = bool(label_pattern.search(window))

        if has_num and has_lbl:
            # Try to anchor the start position sanely
            if re.match(rf"^\s*{number}\.\s*", ln):
                start_pos = i - 1 if (i > 0 and re.search(r"Duties$", prev, re.I)) else i
                next_idx  = i + 1
                return start_pos, "", next_idx
            if i > 0 and re.match(rf"^\s*{number}\.\s*$", prev):
                return i - 1, "", i
            if i + 1 < len(lines) and re.match(rf"^\s*{number}\.\s*", nxt):
                return i, "", i + 2

        if is_all_caps_title(ln):
            return None, "", start_idx
        i += 1

    return None, "", start_idx


    header_codes = collect_codes_around_title(lines, title_idx, title_end_idx, is_sdi_block)

    # fallback if SDI not found near title text
    if is_sdi_block and not header_codes and title_sdi_code:
        header_codes = [title_sdi_code]

    # DEBUG: show each block's title and how many header codes we found
    print(f"[DBG] Title='{specialty_title}'  SDI={is_sdi_block}  codes={header_codes[:6]}")
    rows_before = len(rows)

    if is_sdi_block:
        fanout_codes = header_codes          # one SDI like 8P100
    else:
        fanout_codes = [c for c in header_codes if skill_from_code(c) in {"3","5","7","9"}]


    after = lines[title_end_idx:]

    # --- DIAGNOSTIC START ---
    def _peek(arr, n=18):
        return [f"{i:02d}: {arr[i]}" for i in range(min(n, len(arr)))]

    # try to find section headers using your function (don’t change logic)
    idx = 0
    s1_pos, s1_header_rest, idx_after_1 = find_section_header(after, idx, 1, SECTION_1_RE, LABEL_1_ONLY_RE)
    s2_pos, s2_header_rest, idx_after_2 = find_section_header(after, idx, 2, SECTION_2_RE, LABEL_2_ONLY_RE)
    q_pos,  _s3_hdr_rest, idx_after_3   = find_section_header(after, idx, 3, SECTION_3_RE, LABEL_3_ONLY_RE)

    if not is_sdi_block:
        if s1_pos is None or s2_pos is None:   # only dump when something failed
            print(f"[DBG-LINES] AFTER '{specialty_title}' (p{pstart}-{pend})")
            for line in _peek(after, 22):
                print("   ", line)
            print(f"[DBG-FIND] s1_pos={s1_pos}  s2_pos={s2_pos}  q_pos={q_pos}")
    # keep existing variables for the rest of your logic:
    # (comment out the duplicate calls below if you had them later)
    # --- DIAGNOSTIC END ---

    # --- Section 1: Summary ---
    idx = 0
    s1_pos, s1_header_rest, idx_after_1 = find_section_header(after, idx, 1, SECTION_1_RE, LABEL_1_ONLY_RE)
    summary_text = ""
    if s1_pos is not None:
        idx = idx_after_1
        body_text, consumed = collect_until(
            after[idx:], lambda s: (
                SECTION_2_RE.search(s) or LABEL_2_ONLY_RE.search(s) or
                SECTION_3_RE.search(s) or LABEL_3_ONLY_RE.search(s) or
                is_all_caps_title(s)
            )
        )
        summary_text = norm_ws((s1_header_rest + " " + body_text).strip())
        idx += consumed

        for code in fanout_codes:
            if is_sdi_block:
                lvl, lvl_name = "", ""
                afsc_plain    = code
                major2 = family3 = ""
                search_key     = afsc_plain
                join_key       = afsc_plain
            else:
                lvl = skill_from_code(code)
                if not lvl or lvl == "1":
                    continue
                lvl_name  = skill_name_map.get(lvl, "")
                afsc_plain = code.rstrip("*")
                major2, family3 = major_from_afsc(code)
                search_key = afsc_plain
                join_key   = f"{afsc_plain}|{lvl}"
            rows.append({
                "row_type":"summary","section":"1","section_index":"1",
                "section_label":"Specialty Summary","section_sort":parse_idx("1"),
                "specialty_title":specialty_title,
                "service":SERVICE,"major_group_code":major2,"afsc_family":family3,
                "afsc_code":code,"afsc_code_plain":afsc_plain,
                "skill_level_num":lvl,"skill_level_name":lvl_name,
                "text":summary_text,"page_start":str(pstart),"page_end":str(pend),
                "effective_date":EFFECTIVE_DATE,"source_file":source_file,
                "search_key":search_key,"join_key_afsc_skill":join_key,
                "id":make_id(afsc_plain,lvl,"summary","1",specialty_title,summary_text),
            })
        print(f"[DBG-EMIT] {specialty_title}: summary rows added = {len(rows)-rows_before}")
        rows_before = len(rows)

    # --- Section 2: Duties ---
    s2_pos, s2_header_rest, idx_after_2 = find_section_header(after, idx, 2, SECTION_2_RE, LABEL_2_ONLY_RE)
    duty_start = idx_after_2 if s2_pos is not None else idx  # if S2 not found, scan from beginning of block
    duty_end   = q_pos if q_pos is not None else len(after)
    duty_rows  = extract_duties(after, duty_start, duty_end)

    print(f"[DBG-DUTY] {specialty_title}: found {len(duty_rows)} duty anchors between {duty_start}..{duty_end}")

    # fan out over AFSC (not SDI)
    fanout_codes = [c for c in header_codes if (not is_sdi_code(c)) and (skill_from_code(c) in {"3","5","7","9"})]
    for code in fanout_codes:
        lvl = skill_from_code(code)
        if not lvl or lvl == "1":
            continue
        lvl_name  = skill_name_map.get(lvl, "")
        afsc_plain = code.rstrip("*")
        major2, family3 = major_from_afsc(code)
        search_key = afsc_plain
        join_key   = f"{afsc_plain}|{lvl}"
        for (lab, heading, txt) in duty_rows:
            rows.append({
                "row_type":"duty","section":"2","section_index":lab,
                "section_label":"Duties and Responsibilities","section_sort":parse_idx(lab),
                "specialty_title":specialty_title,
                "service":SERVICE,"major_group_code":major2,"afsc_family":family3,
                "afsc_code":code,"afsc_code_plain":afsc_plain,
                "skill_level_num":lvl,"skill_level_name":lvl_name,
                "text":txt,"page_start":str(pstart),"page_end":str(pend),
                "effective_date":EFFECTIVE_DATE,"source_file":source_file,
                "search_key":search_key,"join_key_afsc_skill":join_key,
                "id":make_id(afsc_plain,lvl,"duty",lab,specialty_title,txt),
            })
    print(f"[DBG-EMIT] {specialty_title}: duty rows added = {len(rows)-rows_before}")
    rows_before = len(rows)

    # --- Section 3: Experience (Specialty Qualifications) ---
    q_pos, _s3_hdr_rest, idx_after_3 = find_section_header(after, idx, 3, SECTION_3_RE, LABEL_3_ONLY_RE)
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
            print(f"[DBG-EMIT] {specialty_title}: experience rows added = {len(rows)-rows_before}")
            rows_before = len(rows)
    # Section 3 handled above with find_section_spans and '3' in spans

    return rows

# --------- drive the PDF ----------

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
    got_title = False  # <-- track if we've seen the ALL-CAPS specialty title in current block

    def flush_block(pstart, pend, lines):
        if lines:
            all_rows.extend(parse_block(lines, pstart, pend, source_file))

    last_pnum = None
    for (pnum, lines) in pages:
        last_pnum = pnum
        for ln in lines:
            if is_header_line(ln):
                # Only flush if the current block ALREADY had a title.
                # This keeps all the AFSC/CEM header lines together with the title and sections.
                if cur_lines and got_title:
                    flush_block(block_pstart or pnum, pnum, cur_lines)
                    cur_lines = []
                    got_title = False
                    block_pstart = pnum
                elif not cur_lines:
                    block_pstart = pnum
                cur_lines.append(ln)
                continue

            # mark when we hit the ALL-CAPS specialty title
            if is_all_caps_title(ln):
                got_title = True

            if not cur_lines:
                block_pstart = block_pstart or pnum
            cur_lines.append(ln)

    # final flush
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

    # --- High-level row counts ---
    print("Row counts by type:\n", df["row_type"].value_counts())

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

    # --- AFSC-only QA (exclude SDIs which have blank skill_level_num) ---
    non_sdi = df[df["skill_level_num"] != ""]
    if not non_sdi.empty:
        bad = (non_sdi.pivot_table(index=["afsc_code_plain","skill_level_num"],
                                   columns="row_type", values="id",
                                   aggfunc="count", fill_value=0)
                     .query("summary == 0 or duty == 0"))
        if not bad.empty:
            print("Heads-up: some AFSCs missing summary/duty:\n", bad.head(10))

    # --- SDI-only stats ---
    sdi_df = df[df["skill_level_num"] == ""]
    print("SDI rows:", len(sdi_df))
    if not sdi_df.empty:
        print(sdi_df["row_type"].value_counts())
        print(sdi_df.head(10)[["afsc_code_plain","row_type","section","section_index","specialty_title"]].to_string(index=False))
