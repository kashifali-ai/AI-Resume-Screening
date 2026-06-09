"""CSV export for bulk screening results.

Produces the same ranked columns the UI offers as a client-side download, so the
ranking can also be exported programmatically (and unit-tested)."""

import csv
import io

from models import BulkScreeningResponse

_HEADER = [
    "Rank", "Candidate", "Score", "Verdict", "Matched", "Missing",
    "Experience (yr)", "Recommendation", "File",
]


def bulk_to_csv(resp: BulkScreeningResponse) -> str:
    """Render a BulkScreeningResponse as CSV text, rows in ranked order."""
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(_HEADER)
    for i, r in enumerate(resp.results, 1):
        w.writerow([
            i,
            r.candidate_name or "",
            r.score if r.ok else "",
            r.verdict if r.ok else "FAILED",
            r.matched_count if r.ok else "",
            r.missing_count if r.ok else "",
            r.experience_years if r.ok else "",
            (r.recommendation if r.ok else r.error) or "",
            r.filename,
        ])
    return out.getvalue()
