"""Recover blank context passages from their source links with a real browser."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from inspect_non_passage import PROJECT_ROOT, find_non_passage_contexts


CONTENT_SELECTOR = "#divContentDoc"
DEFAULT_CONTEXTS = PROJECT_ROOT / "selected-contexts" / "selected-contexts"
BOT_VERIFICATION_MARKERS = (
    "just a moment",
    "performing security verification",
    "verifying you are human",
    "cf-chl-",
)


def normalize_passage(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def is_bot_verification(title: str, body: str) -> bool:
    page = f"{title}\n{body}".casefold()
    return any(marker in page for marker in BOT_VERIFICATION_MARKERS)


def save_recovered_context(source: Path, destination: Path, passage: str) -> None:
    context = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(context, dict):
        raise ValueError(f"{source} must contain a JSON object")
    context["passage"] = passage
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(context, ensure_ascii=False, indent=4), encoding="utf-8"
    )
    temporary.replace(destination)


def has_passage(path: Path) -> bool:
    if not path.exists():
        return False
    context = json.loads(path.read_text(encoding="utf-8-sig"))
    return isinstance(context, dict) and bool(str(context.get("passage") or "").strip())


def crawl(args: argparse.Namespace) -> int:
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as error:
        raise RuntimeError(
            "Playwright is required: python -m pip install -r codebase/requirements.txt"
        ) from error

    _, missing = find_non_passage_contexts(args.contexts)
    targets = [item for item in missing if item["link"]]
    if args.limit is not None:
        targets = targets[: args.limit]

    output_dir = args.contexts if args.in_place else args.output
    recovered = skipped = failed = 0
    with sync_playwright() as playwright:
        launch = {"headless": args.headless}
        if args.browser != "chromium":
            launch["channel"] = args.browser
        browser = playwright.chromium.launch(**launch)
        page = browser.new_page()
        try:
            for index, item in enumerate(targets, 1):
                source = args.contexts / item["file"]
                destination = output_dir / item["file"]
                if not args.force and has_passage(destination):
                    skipped += 1
                    print(f"[{index}/{len(targets)}] skip {item['document_id']}: already recovered")
                    continue

                print(f"[{index}/{len(targets)}] crawl {item['document_id']}: {item['link']}")
                try:
                    page.goto(
                        item["link"],
                        wait_until="domcontentloaded",
                        timeout=args.timeout * 1000,
                    )
                    content = page.locator(CONTENT_SELECTOR)
                    verification_wait = min(args.timeout, 10)
                    try:
                        content.wait_for(
                            state="visible", timeout=verification_wait * 1000
                        )
                    except PlaywrightError:
                        body = page.locator("body").inner_text()
                        if is_bot_verification(page.title(), body):
                            raise ValueError(
                                "blocked by Cloudflare bot verification; use an "
                                "authorized source/API (verification bypass is not attempted)"
                            )
                        content.wait_for(
                            state="visible",
                            timeout=max(args.timeout - verification_wait, 1) * 1000,
                        )
                    passage = normalize_passage(content.inner_text())
                    if len(passage) < args.minimum_length:
                        raise ValueError(
                            f"extracted only {len(passage)} characters from {page.title()}"
                        )
                    save_recovered_context(source, destination, passage)
                    recovered += 1
                    print(f"  saved {len(passage):,} characters to {destination}")
                except (OSError, ValueError, PlaywrightError) as error:
                    failed += 1
                    print(f"  failed: {error}", file=sys.stderr)
                if index < len(targets):
                    time.sleep(args.delay)
        finally:
            browser.close()

    print(f"Recovered: {recovered}; skipped: {skipped}; failed: {failed}")
    return 1 if failed else 0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, default=DEFAULT_CONTEXTS)
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "recovered-contexts"
    )
    output.add_argument(
        "--in-place", action="store_true", help="replace blank source context files"
    )
    parser.add_argument("--browser", choices=("msedge", "chrome", "chromium"), default="msedge")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--timeout", type=int, default=120, help="seconds per page")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between pages")
    parser.add_argument("--minimum-length", type=int, default=200)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true", help="recrawl existing outputs")
    args = parser.parse_args()

    if args.timeout <= 0 or args.delay < 0 or args.minimum_length <= 0:
        parser.error("timeout and minimum-length must be positive; delay cannot be negative")
    try:
        raise SystemExit(crawl(args))
    except (OSError, json.JSONDecodeError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
