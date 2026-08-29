"""Show the last N requests joined to their async eval results.

    python manage.py dump_eval            # last 20 requests + eval (if any)
    python manage.py dump_eval -n 5
    python manage.py dump_eval --pending  # only requests with no eval row yet

Joins rag_request_metrics <- eval_results on request_id (two tables, two
writers, one join - see the models).
"""

from django.core.management.base import BaseCommand

from rag_engine.models import EvalResult, RequestMetric

COLUMNS = [
    ("request_id",  lambda m, e: str(m.request_id)[:8],                          8),
    ("when",        lambda m, e: m.timestamp.strftime("%H:%M:%S"),               8),
    ("total_ms",    lambda m, e: _n(m.total_time_ms),                            9),
    ("tok/s",       lambda m, e: _n(m.tokens_per_second, 1),                     7),
    ("eval?",       lambda m, e: "yes" if e else "-",                            5),
    ("gt?",         lambda m, e: ("y" if e.matched_ground_truth else "n") if e else "-", 3),
    ("kw_src",      lambda m, e: e.keyword_source if e else "-",               12),
    ("kw_cov",      lambda m, e: _pct(e.keyword_coverage) if e else "-",         7),
    ("hit",         lambda m, e: _hit(e) if e else "-",                          4),
    ("rr",          lambda m, e: _n(e.reciprocal_rank, 2) if e else "-",         5),
    ("target",      lambda m, e: (e.target_section if e and e.target_section else "-"), 10),
    ("query",       lambda m, e: " ".join(m.query_text.split())[:38],          38),
]


def _n(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


def _pct(v):
    return "-" if v is None else f"{v * 100:.0f}%"


def _hit(e):
    return "-" if e.hit is None else ("Y" if e.hit else "N")


class Command(BaseCommand):
    help = "Dump the last N requests joined to their async eval results."

    def add_arguments(self, parser):
        parser.add_argument("-n", "--limit", type=int, default=20)
        parser.add_argument("--pending", action="store_true",
                            help="only requests that have no eval row yet")

    def handle(self, *args, **opts):
        metrics = list(
            RequestMetric.objects.order_by("-timestamp")[: opts["limit"]]
        )
        metrics.reverse()

        evals = {
            str(e.request_id): e
            for e in EvalResult.objects.filter(
                request_id__in=[m.request_id for m in metrics]
            )
        }

        if opts["pending"]:
            metrics = [m for m in metrics if str(m.request_id) not in evals]

        if not metrics:
            self.stdout.write("nothing to show")
            return

        headers = [c[0] for c in COLUMNS]
        widths = [c[2] for c in COLUMNS]
        self.stdout.write(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
        self.stdout.write("-+-".join("-" * w for w in widths))
        joined = 0
        for m in metrics:
            e = evals.get(str(m.request_id))
            joined += 1 if e else 0
            cells = [fn(m, e).ljust(w)[:w] for (_, fn, w) in COLUMNS]
            self.stdout.write(" | ".join(cells))

        self.stdout.write("")
        self.stdout.write(
            f"{len(metrics)} requests | {joined} with eval_results | "
            f"{len(metrics) - joined} pending"
        )
