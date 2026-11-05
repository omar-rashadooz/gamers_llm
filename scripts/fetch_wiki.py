import argparse, re, time
from collections import deque
from pathlib import Path
import requests
from textclean import clean_line

API = "https://en.wikipedia.org/w/api.php"
# Wikimedia asks for a descriptive User-Agent with a way to contact you.
# Put your own email here (it is only sent to Wikipedia).
USER_AGENT = "GameLM-learning-project/0.1 (personal learning project; contact: mokh51683@gmail.com)"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
PAUSE = 0.5  # seconds between page downloads

CATEGORIES = [
    "Video game genres", "Video game gameplay", "Video game design",
    "Video game development", "Game design", "Role-playing video games",
    "Computer programming", "Programming paradigms",
    "Programming language topics", "Software engineering",
    "Computer science", "Algorithms", "Data structures",
    # added for the bigger download (a name that does not exist is simply skipped)
    "Video game franchises", "Video game industry", "Video game platforms",
    "Video game terminology", "Video game engines", "Esports",
    "Programming languages", "Programming tools", "Software development",
]
STOP = {"references", "see also", "external links", "notes",
        "further reading", "bibliography", "sources", "citations", "footnotes"}

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def api(params):
    params = {**params, "format": "json"}
    for attempt in range(1, 6):
        try:
            r = session.get(API, params=params, timeout=30)
            if r.status_code == 429:
                wait = min(int(r.headers.get("Retry-After", 30)), 120)
                print(f"  rate limited (429), waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            print(f"  request failed ({e.__class__.__name__}), retry {attempt}/5", flush=True)
            time.sleep(2 * attempt)
    return None


def members(cat, kind):
    out, cont = [], {}
    while True:
        d = api({"action": "query", "list": "categorymembers",
                 "cmtitle": "Category:" + cat, "cmtype": kind,
                 "cmlimit": "500", **cont})
        if not d:
            break
        out += [m["title"] for m in d["query"]["categorymembers"]]
        if "continue" not in d:
            break
        cont = d["continue"]
        time.sleep(PAUSE)
    return out


def collect_titles(depth, limit):
    titles, seen = [], set()
    seen_cats = set(CATEGORIES)
    frontier = [(c, 0) for c in CATEGORIES]
    while frontier and len(titles) < limit:
        cat, d = frontier.pop(0)
        print(f"  [{len(titles)} titles] {cat}", flush=True)
        for t in members(cat, "page"):
            if t not in seen and not t.startswith("List of"):
                seen.add(t)
                titles.append(t)
        if d < depth:
            for sc in members(cat, "subcat"):
                sc = sc.removeprefix("Category:")
                if sc not in seen_cats:
                    seen_cats.add(sc)
                    frontier.append((sc, d + 1))
        time.sleep(PAUSE)
    return titles


def fetch_text(title):
    """Returns the article text, or None if the request failed."""
    d = api({"action": "query", "prop": "extracts", "explaintext": 1,
             "titles": title, "redirects": 1})
    if not d:
        return None
    return next(iter(d["query"]["pages"].values())).get("extract", "")


def clean(text):
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        m = re.match(r"^=+\s*(.*?)\s*=+$", s)
        if m:
            if m.group(1).lower() in STOP:
                break
            continue
        s = clean_line(re.sub(r"\[\d+\]", "", s))
        if s is None or len(s) < 40 or "{" in s or "}" in s or s.endswith(";"):
            continue
        lines.append(s)
    return "\n".join(lines)


def fmt_time(seconds):
    """Seconds -> '1h 05m', '12m 30s' or '45s'."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def eta_text(stamps, remaining):
    """Estimate the time left from the CURRENT speed.

    `stamps` holds the start time of the last ~50 pages, so the speed
    reflects recent progress (including rate-limit waits), not the
    average since the beginning. `remaining` is the number of pages left.
    """
    if len(stamps) < 2:
        return "ETA: calculating..."
    elapsed = stamps[-1] - stamps[0]
    if elapsed <= 0:
        return "ETA: calculating..."
    speed = (len(stamps) - 1) / elapsed  # pages per second
    secs = remaining / speed
    finish = time.strftime("%H:%M", time.localtime(time.time() + secs))
    return f"{speed * 60:.0f} pages/min | left: {fmt_time(secs)} | done ~{finish}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=30)
    ap.add_argument("--depth", type=int, default=1)
    a = ap.parse_args()

    if "YOUR_EMAIL_HERE" in USER_AGENT:
        print("Tip: edit USER_AGENT at the top of this file and put your email.")

    RAW.mkdir(parents=True, exist_ok=True)
    done_file = RAW / "done_titles.txt"
    done = set(done_file.read_text().splitlines()) if done_file.exists() else set()

    print("Collecting titles...")
    titles = [t for t in collect_titles(a.depth, a.max_pages + len(done))
              if t not in done]
    titles = titles[: a.max_pages]
    print(f"{len(titles)} pages to fetch")

    chars, failed = 0, 0
    recent = deque(maxlen=50)  # start times of the last 50 pages
    with open(RAW / "wiki.txt", "a", encoding="utf-8") as out, \
         open(done_file, "a", encoding="utf-8") as df:
        for i, t in enumerate(titles, 1):
            recent.append(time.time())
            raw = fetch_text(t)
            if raw is None:      # failed: don't mark as done, retry next run
                failed += 1
                continue
            text = clean(raw)
            if len(text) >= 500:
                out.write(text + "\n<|endoftext|>\n")
                out.flush()
                chars += len(text)
            df.write(t + "\n")
            df.flush()
            if i % 10 == 0:
                print(f"{i}/{len(titles)}  chars: {chars:,}  | "
                      f"{eta_text(recent, len(titles) - i)}", flush=True)
            time.sleep(PAUSE)
    print(f"Done. New chars: {chars:,}  | failed (will retry next run): {failed}")


if __name__ == "__main__":
    main()
