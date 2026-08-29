"""Create / list gateway API keys.

    python manage.py create_api_key --owner "Acme Corp"
    python manage.py create_api_key --owner "Load test" --rpm 5
    python manage.py create_api_key --list
"""

from django.core.management.base import BaseCommand, CommandError

from rag_engine.models import ApiKey


class Command(BaseCommand):
    help = "Create a new API key (or --list existing ones)."

    def add_arguments(self, parser):
        parser.add_argument("--owner", help="owner / team name for the new key")
        parser.add_argument("--rpm", type=int, default=None,
                            help="per-key requests/minute limit (default: project default)")
        parser.add_argument("--inactive", action="store_true",
                            help="create the key disabled")
        parser.add_argument("--list", action="store_true", help="list all keys and exit")

    def handle(self, *args, **opts):
        if opts["list"]:
            rows = ApiKey.objects.all()
            if not rows:
                self.stdout.write("no API keys")
                return
            for k in rows:
                self.stdout.write(
                    f"{k.key}  owner={k.owner!r}  active={k.is_active}  "
                    f"rpm={k.requests_per_minute or 'default'}  created={k.created_at:%Y-%m-%d}"
                )
            return

        if not opts["owner"]:
            raise CommandError("--owner is required (or pass --list)")

        key = ApiKey.objects.create(
            owner=opts["owner"],
            requests_per_minute=opts["rpm"],
            is_active=not opts["inactive"],
        )
        self.stdout.write(self.style.SUCCESS("API key created:"))
        self.stdout.write(f"  key   : {key.key}")
        self.stdout.write(f"  owner : {key.owner}")
        self.stdout.write(f"  rpm   : {key.requests_per_minute or 'project default'}")
        self.stdout.write(f"  active: {key.is_active}")
        self.stdout.write("\nSend it as:  -H 'X-API-Key: %s'" % key.key)
