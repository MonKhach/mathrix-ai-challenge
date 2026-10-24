"""Terminal output: time, speaker, Armenian original and English translation."""


def format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def format_segment(offset_s: float, label: str, armenian: str, english: str,
                   latency_s: float | None = None, similarity: float | None = None,
                   confidence: float | None = None, timing: dict | None = None) -> str:
    """timing (debug, M5): name -> seconds, e.g. {"phrase": 4.2, "wait": 0.7, "asr": 0.2}."""
    header = f"[{format_timestamp(offset_s)}] {label}"
    if latency_s is not None:
        header += f"   (latency {latency_s:.1f} s)"
    if similarity is not None:
        header += f"   (similarity {similarity:.2f})"
    if confidence is not None:
        header += f"   (confidence {confidence:.2f})"
    if timing:
        header += "   (" + ", ".join(f"{name} {s:.1f} s" for name, s in timing.items()) + ")"
    return f"{header}\nHY: {armenian}\nEN: {english}\n"
