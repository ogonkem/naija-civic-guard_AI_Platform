"""Ingest the constitution PDF into ChromaDB (vector index).

    python manage.py ingest                 # (re)build the index
    python manage.py ingest --if-empty      # skip if the collection has docs

`--if-empty` is what the container entrypoint runs on boot so a fresh
`docker compose up` bootstraps its own ChromaDB.
"""

from django.core.management.base import BaseCommand

from rag_engine.chroma import COLLECTION_NAME, get_chroma_client


class Command(BaseCommand):
    help = "Ingest the constitution PDF into ChromaDB."

    def add_arguments(self, parser):
        parser.add_argument("--if-empty", action="store_true",
                            help="skip when the collection already has documents")
        parser.add_argument(
            "--pdf",
            default="constitution-of-the-federal-republic-of-nigeria.pdf",
            help="path to the constitution PDF",
        )

    def handle(self, *args, **opts):
        if opts["if_empty"]:
            try:
                n = get_chroma_client().get_collection(COLLECTION_NAME).count()
            except Exception:
                n = 0
            if n > 0:
                self.stdout.write(f"ChromaDB collection already has {n} chunks — skipping ingest")
                return

        from ingest import ConstitutionIngestor  # repo-root module

        self.stdout.write("Ingesting constitution PDF into ChromaDB…")
        ConstitutionIngestor(opts["pdf"]).process()
        self.stdout.write(self.style.SUCCESS("Ingest complete."))
