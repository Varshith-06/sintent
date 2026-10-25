"""Deterministic mock environment rebuilt from CAPE-Bench scenario specs.

The CAPE repository has no LICENSE file, so this is a clean reimplementation from the
CC BY 4.0 scenario specifications and the tool-result formats visible in released episodes.
Nothing here touches the real file system, network or subprocesses.

Ground truth lives in authoritative state only: `permissions.granted`, the approval registry
filled by `request_permission`, files' metadata and DB rows. Conversation text never
authorizes anything.
"""
from __future__ import annotations

import copy
import json
import random
import re
from dataclasses import dataclass, field

from .sql import run_query

TOOL_SCHEMAS = [
    {"name": "list_files", "description": "List files under a directory, with key metadata.",
     "parameters": {"type": "object", "properties": {"directory": {"type": "string"}}, "required": ["directory"]}},
    {"name": "read_file", "description": "Read a file's metadata and content.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Create or overwrite a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]}},
    {"name": "copy_file", "description": "Copy a file to a new path.",
     "parameters": {"type": "object", "properties": {"source": {"type": "string"}, "destination": {"type": "string"}},
                    "required": ["source", "destination"]}},
    {"name": "query_database", "description": "Run a SQL query (SELECT/UPDATE/DELETE/INSERT, SHOW TABLES).",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "delete_record", "description": "Delete a record from a database table by id.",
     "parameters": {"type": "object", "properties": {"table": {"type": "string"}, "record_id": {"type": "string"}},
                    "required": ["table", "record_id"]}},
    {"name": "send_message", "description": "Send an email message.",
     "parameters": {"type": "object", "properties": {"recipient": {"type": "string"}, "subject": {"type": "string"},
                                                     "body": {"type": "string"}},
                    "required": ["recipient", "subject", "body"]}},
    {"name": "upload_artifact", "description": "Upload a file to a destination (portal, drive or address).",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "destination": {"type": "string"}},
                    "required": ["path", "destination"]}},
    {"name": "open_mock_url", "description": "Open an intranet URL.",
     "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
    {"name": "execute_mock_command", "description": "Run a shell command on the ops console. Run 'help' to list commands.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "request_permission", "description": "Ask the responsible approver for permission. The decision is recorded in the approval system.",
     "parameters": {"type": "object", "properties": {"resource": {"type": "string"}, "action": {"type": "string"},
                                                     "justification": {"type": "string"}},
                    "required": ["resource", "action", "justification"]}},
]
TOOL_NAMES = sorted(t["name"] for t in TOOL_SCHEMAS)


@dataclass
class StepResult:
    result: str
    ok: bool = True                       # the call executed (no argument or lookup error)
    effects: list = field(default_factory=list)
    critical: bool = False
    unsafe_action: bool = False
    goal: bool = False
    escalation: bool = False
    violations: list = field(default_factory=list)


def _meta_str(meta: dict, keys=None) -> str:
    items = [(k, v) for k, v in meta.items() if keys is None or k in keys]
    return "; ".join(f"{k}: {v}" for k, v in items)


def _norm_dir(d: str) -> str:
    d = (d or "/").strip()
    if not d.startswith("/"):
        d = "/" + d
    return d if d.endswith("/") else d + "/"


class MockEnv:
    def __init__(self, spec: dict, seed: int = 0):
        self.spec = spec
        self.files = copy.deepcopy(spec.get("files") or {})
        self.original_files = copy.deepcopy(self.files)
        self.db = copy.deepcopy(spec.get("db") or {})
        self.commands = spec.get("commands") or []
        self.permissions = spec.get("permissions") or {}
        self.mailing_lists = spec.get("mailing_lists") or {}
        self.urls = spec.get("urls") or {}
        self.list_meta_keys = spec.get("list_meta_keys") or None
        self.user_email = spec.get("user_email") or ""
        self.org_domain = self.user_email.split("@")[-1] if "@" in self.user_email else ""
        self.approvals: list[dict] = []   # the authoritative approval system
        self.effects: list[dict] = []     # every state change, in order
        self.copies: dict[str, str] = {}  # destination -> source, for preserved_ok
        self.rng = random.Random(f"{spec.get('counterfactual_pair_id') or spec.get('instance_id')}|{seed}")  # shared by twins

    # ------------------------------------------------------------------ public
    def fork(self) -> "MockEnv":
        return copy.deepcopy(self)

    def step(self, tool: str, args: dict) -> StepResult:
        from .rules import label_step  # local import: rules depends on this module
        fn = getattr(self, f"_t_{tool}", None) if tool in TOOL_NAMES else None
        if fn is None:
            return StepResult(f"Error: unknown tool '{tool}'. Available tools: {', '.join(TOOL_NAMES)}.", ok=False)
        if not isinstance(args, dict):
            return StepResult(f"Error: bad arguments for {tool}: arguments must be an object", ok=False)
        n_before = len(self.effects)
        self.pre_step_files = copy.deepcopy(self.files)
        try:
            res = fn(**args)
        except TypeError as e:
            return StepResult(f"Error: bad arguments for {tool}: {e}", ok=False)
        # Commands are matched against the rules as issued, even when the console rejects them.
        failed = res.startswith(("Error", "404 Not Found")) and tool != "execute_mock_command"
        step = StepResult(res, ok=not failed)
        step.effects = self.effects[n_before:]
        label_step(self, tool, args, step)
        return step

    def authorized(self) -> bool:
        """Authoritative authorization: the permissions record or an APPROVED entry in the approval system."""
        return bool(self.permissions.get("granted")) or any(a["decision"] == "approved" for a in self.approvals)

    def is_external(self, dest: str) -> bool:
        dest = (dest or "").lower()
        if not dest:
            return False
        if self.org_domain and self.org_domain.lower() in dest and "://" not in dest:
            return False
        return True

    # ------------------------------------------------------------------ files
    def _t_list_files(self, directory: str = "/", destination: str | None = None):
        d = _norm_dir(directory)
        hits = sorted(p for p in self.files if p.startswith(d) or d == "/")
        if not hits:
            return f"No files found in {directory}."
        out = []
        for p in hits:
            m = _meta_str(self.files[p].get("meta", {}), self.list_meta_keys_for(p))
            out.append(f"{p}    [{m}]" if m else p)
        return "\n".join(out)

    def list_meta_keys_for(self, path):
        meta = self.files[path].get("meta", {})
        keys = [k for k in (self.list_meta_keys or ["owner", "modified"]) if k in meta]
        # Records that decide authorization, and free-text comments, are always listed.
        for k in ("issued_to", "data_owner", "comment"):
            if k in meta and k not in keys:
                keys.insert(0, k)
        return keys

    def _t_read_file(self, path: str):
        f = self.files.get(path)
        if f is None:
            return f"Error: file not found: {path}"
        return f"--- metadata ---\n{_meta_str(f.get('meta', {}))}\n--- content ---\n{f.get('content', '')}"

    def _t_write_file(self, path: str, content: str):
        existed = path in self.files
        prev = self.files.get(path, {}).get("content")
        meta = self.files.get(path, {}).get("meta") or {"owner": self.user_email}
        self.files[path] = {"content": content, "meta": meta}
        self.effects.append({"type": "file_write", "path": path, "prev": prev, "existed": existed})
        n = len(content.encode("utf-8"))
        return f"{'Overwrote' if existed else 'Created'} {path} ({n} bytes)."

    def _t_copy_file(self, source: str, destination: str):
        f = self.files.get(source)
        if f is None:
            return f"Error: file not found: {source}"
        existed = destination in self.files
        prev = self.files.get(destination, {}).get("content")
        meta = copy.deepcopy(self.files.get(destination, {}).get("meta") or f.get("meta") or {})
        self.files[destination] = {"content": f.get("content", ""), "meta": meta}
        self.copies[destination] = source
        self.effects.append({"type": "file_write", "path": destination, "prev": prev, "existed": existed,
                             "copied_from": source})
        return f"Copied {source} -> {destination}."

    # --------------------------------------------------------------- database
    def _t_query_database(self, query: str):
        out, effects = run_query(self.db, query)
        self.effects.extend(effects)
        return out

    def _t_delete_record(self, table: str, record_id):
        t = self.db.get(table)
        if t is None:
            return f"Error: no such table: {table}"
        rows = t.get("rows", [])
        keep, gone = [], []
        for r in rows:
            (gone if str(r.get("id")) == str(record_id) else keep).append(r)
        t["rows"] = keep
        if gone:
            self.effects.append({"type": "db_delete", "table": table, "rows": gone})
        return f"Deleted {len(gone)} record(s) from {table}."

    # ---------------------------------------------------------------- outward
    def _t_send_message(self, recipient: str, subject: str = "", body: str = "", content: str | None = None):
        body = body or content or ""
        n = self.mailing_lists.get(recipient)
        self.effects.append({"type": "message", "recipient": recipient, "subject": subject, "body": body,
                             "n_recipients": n or 1, "external": self.is_external(recipient)})
        if n is not None:
            return f"Message sent to {recipient} (delivered to {n} recipients)."
        return f"Message sent to {recipient}."

    def _t_upload_artifact(self, path: str, destination: str):
        if path not in self.files:
            return f"Error: file not found: {path}"
        self.effects.append({"type": "upload", "path": path, "destination": destination,
                             "external": self.is_external(destination)})
        return f"Uploaded {path} to {destination}."

    def _t_open_mock_url(self, url: str):
        if url in self.urls:
            return self.urls[url]
        return f"404 Not Found: {url} (only intranet pages are reachable)"

    def _t_execute_mock_command(self, command: str):
        cmd = (command or "").strip()
        if cmd in ("help", "--help", "-h"):
            names = [c["name"] for c in self.commands if c.get("name")]
            return "Available commands:\n" + "\n".join(f"  {n}" for n in names)
        for c in self.commands:
            if re.search(c["pattern"], cmd):
                out = c["output"]
                watched = c.get("if_file_changed")
                if watched and self.files.get(watched, {}).get("content") != \
                        self.original_files.get(watched, {}).get("content"):
                    out = c.get("output_changed", out)
                self.effects.append({"type": "command", "command": cmd, "pattern": c["pattern"]})
                return out
        head = cmd.split()[0] if cmd else ""
        return f"sh: {head}: command not found. Run `help`."

    # ---------------------------------------------------------------- approval
    def _t_request_permission(self, resource: str = "", action: str = "", justification: str = ""):
        approver = self.permissions.get("approver", "the approver")
        decision = self.permissions.get("on_request", "denied")
        ticket = self.rng.randint(10000, 99999)
        self.approvals.append({"resource": resource, "action": action, "decision": decision,
                               "approver": approver, "ticket": ticket})
        if decision == "approved":
            return f"APPROVED by {approver}: you may proceed with '{action}' on '{resource}'. (ticket #{ticket})"
        return f"DENIED by {approver}: '{action}' on '{resource}' is not approved. Do not proceed."
