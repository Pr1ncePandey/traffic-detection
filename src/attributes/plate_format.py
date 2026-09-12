"""Plate text post-processing, shared by every OCR backend.

Applied in order by the caller: join_rows() -> normalize() -> correct().

Why formats help: in the *series* part of an Indian registration the letters
'I' and 'O' are never issued, precisely so they cannot be confused with '1'
and '0'. Every slot's character class is therefore known up front, so a
character that violates its slot is almost certainly an OCR slip and can be
mapped back. Note the exclusion is a property of the series slots only - the
state code 'OD' (Odisha) legitimately starts with an O.

Honest limit worth knowing before trusting this: it only repairs CROSS-class
slips, where a digit was read in a letter slot or vice versa (3<->D, 0<->O,
5<->S ...). Two of the three confusions this project has actually measured,
C<->L and 4<->6, are WITHIN one class - both characters are legal in that
slot, so no format rule can tell them apart. Those need a better recogniser,
which is what swapping plate.ocr_backend is for.

Correction is a prior, not a filter: a string that fits no known format is
returned unchanged rather than dropped, because Indian plates are frequently
non-compliant (odd fonts, extra characters, decorative text).
"""

MIN_PLATE_LEN = 4  # fewer alphanumerics than this is never a plate
# A single cross-class slip is a plausible OCR confusion; two or more means
# the read is probably just wrong, and "repairing" it into a valid-looking
# plate moves further from the truth. Measured on samples/plate_gt.csv (n=5,
# so directional not conclusive): CER with a budget of off/1/2/3 came out
# 0.208/0.188/0.229/0.229 for paddle_anpr and 0.896/0.875/0.896/0.896 for
# rapidocr - a budget of 1 was best for both, and anything looser was worse.
MAX_REPAIRS = 1

# Slot classes used in the templates below.
_STATE = "S"   # state code letter: any A-Z ('OD' Odisha really does start O)
_SERIES = "L"  # series letter: A-Z minus I and O (never issued)
_DIGIT = "D"

_EXCLUDED_SERIES_LETTERS = frozenset("IO")

# Came out a letter where a digit belongs -> the digit it almost certainly is.
_AS_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1",
             "Z": "2", "A": "4", "S": "5", "G": "6", "C": "6", "T": "7",
             "B": "8"}

# Came out a digit where a letter belongs -> the letter it almost certainly is.
# '0' maps to D and '1' to L, NOT to O and I: those two letters are exactly
# the ones Indian series never use, which is what makes this direction
# decidable at all.
_AS_LETTER = {"0": "D", "1": "L", "2": "Z", "3": "D", "4": "A", "5": "S",
              "6": "G", "7": "T", "8": "B", "9": "G"}


def _build_templates():
    """(slot pattern, {index: required literal}) ordered most-common first."""
    out = []
    # Standard: state(2) + RTO district(1-2 digits) + series(0-3) + number(4).
    # A zero-letter series is legal on older plates (e.g. MH121234).
    for n_digits in (2, 1):
        for n_series in (2, 1, 3, 0):
            out.append((_STATE * 2 + _DIGIT * n_digits
                        + _SERIES * n_series + _DIGIT * 4, {}))
    # BH (Bharat) series: YY + literal 'BH' + 4 digits + 1-2 series letters.
    for n_series in (2, 1):
        out.append((_DIGIT * 2 + _SERIES * 2 + _DIGIT * 4
                    + _SERIES * n_series, {2: "B", 3: "H"}))
    # VA (vintage): state + literal 'VA' + 2 series letters + 4 digits.
    out.append((_STATE * 2 + _SERIES * 2 + _SERIES * 2 + _DIGIT * 4,
                {2: "V", 3: "A"}))
    return out


_TEMPLATES = _build_templates()
_LENGTHS = sorted({len(p) for p, _ in _TEMPLATES})


def normalize(text: str) -> str:
    """Uppercase, keep only alphanumerics. The one true normalizer."""
    return "".join(c for c in (text or "").upper() if c.isalnum())


ROW_TOL = 0.6  # vertical gap, in text heights, that starts a new row


def join_rows(parts, row_tol: float = ROW_TOL) -> str:
    """Stitch OCR regions into one plate string, in reading order.

    `parts` is an iterable of (centre_y, centre_x, height, text).

    Group into rows first, then order left-to-right WITHIN each row. Sorting
    on y alone is not enough: a generic OCR backend often splits even a
    single-row plate into two or three regions whose y values are nearly
    equal, so their relative order comes out arbitrary - 'DL12C0705' came back
    as '0705JZL10'. Banding by y and sorting by x inside the band fixes that,
    and still keeps a genuine two-row plate whole, which taking only the
    highest-confidence region never did.
    """
    parts = list(parts)
    if not parts:
        return ""
    heights = sorted(p[2] for p in parts)
    median_h = heights[len(heights) // 2] or 1.0
    rows, band, band_y = [], [], None
    for cy, cx, _h, text in sorted(parts, key=lambda p: p[0]):
        if band_y is not None and (cy - band_y) > row_tol * median_h:
            rows.append(band)
            band = []
        band.append((cx, text))
        band_y = cy if band_y is None else band_y
        if not band[:-1]:
            band_y = cy
    rows.append(band)
    return "".join(text for band in rows
                   for _cx, text in sorted(band, key=lambda b: b[0]))


def looks_like_plate(text: str) -> bool:
    """Loose gate: long enough, and mixes letters with digits."""
    return (len(text) >= MIN_PLATE_LEN
            and any(c.isalpha() for c in text)
            and any(c.isdigit() for c in text))


def _fit(text: str, pattern: str, literals: dict):
    """Coerce `text` into `pattern`. Returns (fitted, n_repairs) or (None, 0)."""
    if len(text) != len(pattern):
        return (None, 0)
    out, repairs = [], 0
    for i, (ch, slot) in enumerate(zip(text, pattern)):
        required = literals.get(i)
        if required is not None:
            if ch != required:
                ch = _AS_LETTER.get(ch, ch)
                if ch != required:
                    return (None, 0)
                repairs += 1
        elif slot == _DIGIT:
            if not ch.isdigit():
                ch = _AS_DIGIT.get(ch)
                if ch is None:
                    return (None, 0)
                repairs += 1
        else:
            if not ch.isalpha():
                ch = _AS_LETTER.get(ch)
                if ch is None:
                    return (None, 0)
                repairs += 1
            if slot == _SERIES and ch in _EXCLUDED_SERIES_LETTERS:
                return (None, 0)
        out.append(ch)
    return ("".join(out), repairs)


def _best_fit(text: str):
    """Fewest-repairs template fit for an exact-length string."""
    best = None
    for pattern, literals in _TEMPLATES:
        fitted, repairs = _fit(text, pattern, literals)
        if fitted is not None and (best is None or repairs < best[1]):
            best = (fitted, repairs)
            if repairs == 0:
                break  # a clean read is never improved on
    return best


def correct(text: str):
    """Repair cross-class OCR slips. Returns (text, matched, n_repairs).

    matched=False means nothing fitted and `text` came back untouched.
    """
    best = _best_fit(text)
    if best is not None and best[1] <= MAX_REPAIRS:
        return (best[0], True, best[1])

    # Surrounding junk: HSRP plates carry an 'IND' band, and a generic OCR
    # backend can read that (or a state name, or a sticker) as another region
    # and join it in. Slide a plate-sized window before giving up, longest
    # first.
    #
    # Windows demand a REPAIR-FREE fit. Guessing where the plate starts is
    # already one assumption; allowing character repairs on top of it turns
    # near-misses into confident wrong answers - e.g. 'MH12IO1234', which must
    # be rejected for I/O in the series, otherwise fits an 8-char window as
    # 'MH121012' with two repairs.
    # An ambiguous window is also declined: 'IND22BH1234AB' contains both
    # 'ND22BH1234' and '22BH1234AB' as clean fits, and choosing between them
    # would be a coin flip dressed up as a correction.
    for length in sorted((n for n in _LENGTHS if n < len(text)), reverse=True):
        hits = set()
        for start in range(len(text) - length + 1):
            fitted = _best_fit(text[start:start + length])
            if fitted is not None and fitted[1] == 0:
                hits.add(fitted[0])
        if len(hits) == 1:
            return (hits.pop(), True, 0)
        if len(hits) > 1:
            return (text, False, 0)

    return (text, False, 0)


def fits_template(text: str) -> bool:
    """True when `text` is a clean fit to a real Indian registration template.

    The gate for keying an IDENTITY on a plate, which is a stricter question
    than looks_like_plate() answers. That one only asks for 4+ alphanumerics
    mixing letters and digits, so the 'IND' band on an HSRP plate joined to a
    couple of digits passes it - fine for displaying a best-effort read, far
    too loose to merge two sightings into one vehicle on.

    Repair-free by design: correct() is allowed MAX_REPAIRS cross-class fixes
    because a displayed plate is better approximately right than blank, but a
    repaired character is a guess, and a guess in an identity key silently
    fuses two vehicles.

    Delegates to correct() rather than _best_fit() so this agrees with what the
    reader actually produced, window slide included: a plate read as
    'INDHR26DK8337' is a clean fit once the HSRP band is dropped, and deciding
    otherwise here would refuse identity to exactly the plates correct() was
    written to rescue.
    """
    _fitted, matched, repairs = correct(normalize(text))
    return bool(matched) and repairs == 0
