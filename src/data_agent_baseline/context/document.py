from __future__ import annotations

from pathlib import Path


class DocumentProcessor:
    def __init__(self, max_chars_per_segment: int = 6000, max_segments: int = 30):
        self.max_chars = max_chars_per_segment
        self.max_segments = max_segments
        self._cache: dict[str, list[str]] = {}

    def segment(self, file_path: Path) -> list[str]:
        cache_key = str(file_path)
        if cache_key in self._cache:
            return self._cache[cache_key]

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        paragraphs = content.split("\n\n")
        segments: list[str] = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) > self.max_chars and current:
                segments.append(current.strip())
                current = para
            else:
                current = current + "\n\n" + para if current else para

        if current.strip():
            segments.append(current.strip())

        segments = segments[:self.max_segments]
        self._cache[cache_key] = segments
        return segments

    def get_segment(self, file_path: Path, index: int) -> str | None:
        segments = self.segment(file_path)
        if 0 <= index < len(segments):
            return f"[Segment {index+1}/{len(segments)}]\n{segments[index]}"
        return None

    def get_segment_count(self, file_path: Path) -> int:
        return len(self.segment(file_path))

    def search_keywords(self, file_path: Path, keywords: list[str]) -> list[dict]:
        segments = self.segment(file_path)
        results = []
        for i, seg in enumerate(segments):
            seg_lower = seg.lower()
            matched = [kw for kw in keywords if kw.lower() in seg_lower]
            if matched:
                results.append({
                    "segment_index": i,
                    "matched_keywords": matched,
                    "preview": seg[:300] + "..." if len(seg) > 300 else seg,
                })
        return results
