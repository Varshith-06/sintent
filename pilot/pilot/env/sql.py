"""A tiny SQL subset over the mock DB: SELECT / UPDATE / DELETE / INSERT / SHOW TABLES.

Tables are {"meta": {...}, "rows": [{...}, ...]}. WHERE supports AND/OR (no parentheses),
=, !=, <>, <, >, <=, >=, LIKE, NOT LIKE, IN (...), IS [NOT] NULL. Unknown columns make a
condition false rather than raising, as in the released episodes.
"""
from __future__ import annotations

import copy
import re

UNSUPPORTED = "Error: unsupported query. Supported: SELECT/UPDATE/DELETE/INSERT, SHOW TABLES."

_VAL = r"('(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|-?\d+(?:\.\d+)?|NULL|TRUE|FALSE)"


def _unquote(v: str):
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        return v[1:-1].replace(v[0] * 2, v[0])
    if v.upper() == "NULL":
        return None
    if v.upper() in ("TRUE", "FALSE"):
        return v.upper() == "TRUE"
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def _cmp(a, b, op):
    if a is None or b is None:
        return False
    try:
        fa, fb = float(a), float(b)
        a, b = fa, fb
    except (TypeError, ValueError):
        a, b = str(a), str(b)
    return {"=": a == b, "==": a == b, "!=": a != b, "<>": a != b,
            "<": a < b, ">": a > b, "<=": a <= b, ">=": a >= b}[op]


def _like(val, pat) -> bool:
    if val is None:
        return False
    rx = "^" + "".join(".*" if c == "%" else "." if c == "_" else re.escape(c) for c in str(pat)) + "$"
    return re.match(rx, str(val), re.IGNORECASE | re.DOTALL) is not None


_COND = re.compile(
    rf"^\s*[`\"]?(\w+)[`\"]?\s*(?:(NOT\s+LIKE|LIKE)\s*{_VAL}|(IS\s+NOT\s+NULL|IS\s+NULL)|"
    rf"(NOT\s+IN|IN)\s*\(([^)]*)\)|(<=|>=|<>|!=|==|=|<|>)\s*{_VAL})\s*$",
    re.IGNORECASE | re.DOTALL)


def _split_top(s: str, word: str):
    """Split on a keyword outside quotes and parentheses."""
    parts, depth, q, cur, i = [], 0, None, "", 0
    w = f" {word} "
    while i < len(s):
        c = s[i]
        if q:
            cur += c
            if c == q:
                q = None
        elif c in "'\"":
            q = c
            cur += c
        elif c == "(":
            depth += 1
            cur += c
        elif c == ")":
            depth -= 1
            cur += c
        elif depth == 0 and s[i:i + len(w)].upper() == w:
            parts.append(cur)
            cur = ""
            i += len(w)
            continue
        else:
            cur += c
        i += 1
    parts.append(cur)
    return parts


def _where(expr: str | None):
    if not expr or not expr.strip():
        return lambda r: True
    ors = [[_cond(c) for c in _split_top(o, "AND")] for o in _split_top(expr.strip(), "OR")]
    if any(c is None for conj in ors for c in conj):
        return None
    return lambda r: any(all(c(r) for c in conj) for conj in ors)


def _cond(text: str):
    t = text.strip()
    while t.startswith("(") and t.endswith(")"):
        t = t[1:-1].strip()
    m = _COND.match(t)
    if not m:
        return None
    col = m.group(1)
    if m.group(2):
        neg, pat = m.group(2).upper().startswith("NOT"), _unquote(m.group(3))
        return lambda r: col in r and (_like(r[col], pat) != neg)
    if m.group(4):
        want_null = "NOT" not in m.group(4).upper()
        return lambda r: col in r and ((r[col] in (None, "")) == want_null)
    if m.group(5):
        neg = m.group(5).upper().startswith("NOT")
        vals = [_unquote(v) for v in re.findall(_VAL, m.group(6))]
        return lambda r: col in r and (any(_cmp(r[col], v, "=") for v in vals) != neg)
    op, val = m.group(7), _unquote(m.group(8))
    return lambda r: col in r and _cmp(r[col], val, op)


def _header(name, table):
    meta = table.get("meta") or {}
    m = "; ".join(f"{k}: {v}" for k, v in meta.items())
    return f"-- table {name} ({m})" if m else f"-- table {name}"


def _no_table(db, name):
    return f"Error: no such table: {name}. Tables: {', '.join(db)}"


def run_query(db: dict, query: str):
    """Returns (output_text, effects). Mutates db in place for UPDATE/DELETE/INSERT."""
    q = (query or "").strip().rstrip(";").strip()
    up = q.upper()
    if re.fullmatch(r"SHOW\s+TABLES", up):
        return "\n".join(db), []

    m = re.match(r"^SELECT\s+(.+?)\s+FROM\s+[`\"]?(\w+)[`\"]?(?:\s+WHERE\s+(.+?))?(?:\s+ORDER\s+BY\s+[\w\s,]+)?$",
                 q, re.IGNORECASE | re.DOTALL)
    if m:
        cols, name, where = m.group(1).strip(), m.group(2), m.group(3)
        if name not in db:
            return _no_table(db, name), []
        pred = _where(where)
        if pred is None:
            return UNSUPPORTED, []
        table = db[name]
        rows = [r for r in table.get("rows", []) if pred(r)]
        if re.fullmatch(r"COUNT\(\s*\*\s*\)(\s+AS\s+\w+)?", cols, re.IGNORECASE):
            return f"{_header(name, table)}\ncount\n{len(rows)}\n(1 rows)", []
        all_cols = list(table["rows"][0].keys()) if table.get("rows") else []
        if cols == "*":
            sel = all_cols
        else:
            sel = [c.strip().strip("`\"") for c in cols.split(",")]
            if not all(re.fullmatch(r"\w+", c) for c in sel):
                return UNSUPPORTED, []
        if not rows:
            return f"{_header(name, table)}\n(0 rows)", []
        lines = [_header(name, table), " | ".join(sel)]
        lines += [" | ".join("" if r.get(c) is None else str(r.get(c)) for c in sel) for r in rows]
        lines.append(f"({len(rows)} rows)")
        return "\n".join(lines), []

    m = re.match(r"^UPDATE\s+[`\"]?(\w+)[`\"]?\s+SET\s+(.+?)(?:\s+WHERE\s+(.+))?$", q, re.IGNORECASE | re.DOTALL)
    if m:
        name, sets, where = m.group(1), m.group(2), m.group(3)
        if name not in db:
            return _no_table(db, name), []
        assigns = []
        for a in re.finditer(rf"[`\"]?(\w+)[`\"]?\s*=\s*({_VAL[1:-1]}|\w+(?:\(\))?)", sets, re.IGNORECASE):
            assigns.append((a.group(1), _unquote(a.group(2))))
        pred = _where(where)
        if not assigns or pred is None:
            return UNSUPPORTED, []
        changed = []
        for r in db[name].get("rows", []):
            if pred(r):
                before = copy.deepcopy(r)
                for c, v in assigns:
                    r[c] = v
                changed.append({"before": before, "after": copy.deepcopy(r)})
        effects = [{"type": "db_update", "table": name, "rows": changed}] if changed else []
        return f"UPDATE {len(changed)}", effects

    m = re.match(r"^DELETE\s+FROM\s+[`\"]?(\w+)[`\"]?(?:\s+WHERE\s+(.+))?$", q, re.IGNORECASE | re.DOTALL)
    if m:
        name, where = m.group(1), m.group(2)
        if name not in db:
            return _no_table(db, name), []
        pred = _where(where)
        if pred is None:
            return UNSUPPORTED, []
        rows = db[name].get("rows", [])
        gone = [r for r in rows if pred(r)]
        db[name]["rows"] = [r for r in rows if not pred(r)]
        effects = [{"type": "db_delete", "table": name, "rows": gone}] if gone else []
        return f"DELETE {len(gone)}", effects

    m = re.match(r"^INSERT\s+INTO\s+[`\"]?(\w+)[`\"]?\s*\(([^)]*)\)\s*VALUES\s*\((.*)\)$", q, re.IGNORECASE | re.DOTALL)
    if m:
        name = m.group(1)
        if name not in db:
            return _no_table(db, name), []
        cols = [c.strip().strip("`\"") for c in m.group(2).split(",")]
        vals = [_unquote(v) for v in re.findall(_VAL, m.group(3))]
        if len(cols) != len(vals):
            return UNSUPPORTED, []
        row = dict(zip(cols, vals))
        db[name].setdefault("rows", []).append(row)
        return "INSERT 1", [{"type": "db_insert", "table": name, "row": row}]

    return UNSUPPORTED, []
