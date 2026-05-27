"""Produce the HMAC headers a gateway must send to legalite-ai.

Useful for:

- Local debugging — sign a request from curl while
  ``LEGALITE_GATEWAY_SECRET`` is set in your env.
- Sanity-checking that legalite-backend's TypeScript signing matches
  this Python primitive byte-for-byte.

Usage::

    # Set the same secret you've configured in Railway:
    export LEGALITE_GATEWAY_SECRET="…"

    # Emit ready-to-paste curl headers:
    python -m scripts.sign_request \\
        --org-id   7f3a4b6c-12ab-4cde-9876-abcdef012345 \\
        --user-id  c91e88f2-aabb-ccdd-eeff-001122334455 \\
        --role     admin \\
        --format   curl

    # Or as a JSON object:
    python -m scripts.sign_request --org-id … --user-id … --role admin --format json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from app.core.service_auth import compute_signature


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--org-id", required=True)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--role", required=True, choices=["owner", "admin", "lawyer", "associate", "viewer"])
    parser.add_argument(
        "--timestamp",
        type=int,
        default=None,
        help="unix seconds (default: now)",
    )
    parser.add_argument(
        "--secret",
        default=None,
        help="override LEGALITE_GATEWAY_SECRET env var",
    )
    parser.add_argument(
        "--format",
        choices=["curl", "json", "headers"],
        default="curl",
        help="output style (default: curl)",
    )
    args = parser.parse_args()

    secret = args.secret or os.environ.get("LEGALITE_GATEWAY_SECRET")
    if not secret:
        print(
            "ERROR: set LEGALITE_GATEWAY_SECRET in the environment or pass --secret",
            file=sys.stderr,
        )
        return 2

    ts = args.timestamp or int(time.time())
    sig = compute_signature(
        timestamp=ts,
        organization_id=args.org_id.lower(),
        user_id=args.user_id.lower(),
        role=args.role.lower(),
        secret=secret,
    )

    headers = {
        "x-organization-id": args.org_id,
        "x-user-id": args.user_id,
        "x-role": args.role,
        "x-legalite-timestamp": str(ts),
        "x-legalite-signature": sig,
    }

    if args.format == "json":
        print(json.dumps(headers, indent=2))
    elif args.format == "headers":
        for k, v in headers.items():
            print(f"{k}: {v}")
    else:  # curl
        parts = ["curl"]
        for k, v in headers.items():
            parts.append(f"-H '{k}: {v}'")
        print(" \\\n  ".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
