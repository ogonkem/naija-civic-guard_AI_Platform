"""Dump the last N logged RAG request metrics as a table.

    python manage.py dump_metrics            # last 20
    python manage.py dump_metrics -n 5       # last 5
    python manage.py dump_metrics --errors   # only failed requests
    python manage.py dump_metrics --csv      # CSV instead of a table
"""

import csv
import sys

from django.core.management.base import BaseCommand

from rag_engine.models import RequestMetric

COLUMNS = [
    ("timestamp",    lambda r: r.timestamp.strftime("%m-%d %H:%M:%S"),   14),
    ("request_id",   lambda r: str(r.request_id)[:8],                     8),
    ("classify",     lambda r: r.classify_label or "-",                  15),
    ("rcalls",       lambda r: _int(r.retrieval_calls),                   6),
    ("retry",        lambda r: "y" if r.verify_retry else "",            5),
    ("clsfy_ms",     lambda r: _num(r.classify_ms),                       9),
    ("retr_ms",      lambda r: _num(r.retrieve_ms),                       9),
    ("chain_ms",     lambda r: _num(r.chain_ms),                          9),
    ("vrfy_ms",      lambda r: _num(r.verify_ms),                         9),
    ("gen_ms",       lambda r: _num(r.generation_time_ms),               10),
    ("total_ms",     lambda r: _num(r.total_time_ms),                    10),
    ("mcp_tool_calls (name latency_ms ok)", lambda r: _tools(r.tool_calls), 46),
    ("tokens",       lambda r: _int(r.tokens_generated),                  7),
    ("error",        lambda r: " ".join((r.error or "").split())[:24],   24),
    ("query",        lambda r: " ".join(r.query_text.split())[:34],      34),
]


def _tools(calls):
    if not calls:
        return "-"
    return " | ".join(
        f"{c['tool_name']} {c['tool_latency_ms']:.1f} {'ok' if c['ok'] else 'ERR'}"
        for c in calls
    )


def _num(v, ndigits=1):
    return "-" if v is None else f"{v:.{ndigits}f}"


def _int(v):
    return "-" if v is None else str(v)


class Command(BaseCommand):
    help = "Dump the last N logged RAG request metrics as a table."

    def add_arguments(self, parser):
        parser.add_argument("-n", "--limit", type=int, default=20,
                            help="number of rows (default 20)")
        parser.add_argument("--errors", action="store_true",
                            help="only rows that recorded an error")
        parser.add_argument("--csv", action="store_true",
                            help="emit CSV (all fields) instead of a table")

    def handle(self, *args, **opts):
        qs = RequestMetric.objects.all()
        if opts["errors"]:
            qs = qs.exclude(error="")
        rows = list(qs.order_by("-timestamp")[: opts["limit"]])
        rows.reverse()  # oldest-first for reading

        if not rows:
            self.stdout.write("no request metrics logged yet")
            return

        if opts["csv"]:
            self._csv(rows)
            return

        self._table(rows)

        gen = [r.generation_time_ms for r in rows if r.generation_time_ms is not None]
        tps = [r.tokens_per_second for r in rows if r.tokens_per_second is not None]
        tot = [r.total_time_ms for r in rows if r.total_time_ms is not None]
        errs = sum(1 for r in rows if r.error)
        self.stdout.write("")
        self.stdout.write(
            f"{len(rows)} rows | errors: {errs} | "
            f"avg total: {_avg(tot)} ms | avg generation: {_avg(gen)} ms | "
            f"avg throughput: {_avg(tps, 1)} tok/s"
        )

    def _table(self, rows):
        headers = [c[0] for c in COLUMNS]
        widths = [c[2] for c in COLUMNS]
        line = "-+-".join("-" * w for w in widths)
        self.stdout.write(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
        self.stdout.write(line)
        for r in rows:
            cells = [fn(r).ljust(w)[:w] for (_, fn, w) in COLUMNS]
            self.stdout.write(" | ".join(cells))

    def _csv(self, rows):
        field_names = [f.name for f in RequestMetric._meta.fields]
        w = csv.writer(sys.stdout)
        w.writerow(field_names)
        for r in rows:
            w.writerow([getattr(r, n) for n in field_names])


def _avg(xs, ndigits=1):
    return "-" if not xs else f"{sum(xs) / len(xs):.{ndigits}f}"
