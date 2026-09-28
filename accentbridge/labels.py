import numpy as np

CV_ACCENTS = {
    "United States English": "us",
    "England English": "england",
    "India and South Asia (India, Pakistan, Sri Lanka)": "india",
    "Canadian English": "canada",
    "Australian English": "australia",
    "Southern African (South Africa, Zimbabwe, Namibia)": "southern_africa",
    "Scottish English": "scotland",
    "New Zealand English": "new_zealand",
    "Irish English": "ireland",
    "Northern Irish": "northern_ireland",
    "Filipino": "philippines",
    "Hong Kong English": "hong_kong",
    "Singaporean English": "singapore",
    "Malaysian English": "malaysia",
    "Welsh English": "wales",
    "West Indies and Bermuda (Bahamas, Bermuda, Jamaica, Trinidad)": "west_indies",
}
HELDOUT = ("australia", "ireland", "philippines")
OOD = ("wales", "west_indies")
EXCLUDE = ("northern_ireland",)
UNSEEN = set(HELDOUT) | set(OOD)

ALIASES = {"en-indian": "indian", "en-american": "american", "en-british": "british", "en-scottish": "scottish",
           "en-filipino": "filipino", "en-singaporean": "singaporean", "en-australian": "australian",
           "us": "american", "uk": "british", "england": "british", "aussie": "australian"}
SUBACCENT_PARENT = {"us southern": "american", "us midwest": "american", "us northeast": "american",
                    "new york": "american", "boston": "american", "chicago": "american", "african american": "american",
                    "received pronunciation": "british", "cockney": "british", "yorkshire": "british",
                    "geordie": "british", "scouse": "british"}
BANK_TO_UNIFIED = {"american": "us", "british": "england", "indian": "india", "australian": "australia",
                   "canadian": "canada", "irish": "ireland", "scottish": "scotland", "welsh": "wales",
                   "south african": "southern_africa", "new zealand": "new_zealand", "filipino": "philippines",
                   "singaporean": "singapore", "jamaican": "west_indies", "malaysian": "malaysia",
                   "hong kong": "hong_kong"}


# Map a Common Voice accent string to a unified label (None if missing or ambiguous).
def cv_accent(s):
    if not isinstance(s, str) or not s:
        return None
    keys = {CV_ACCENTS[p.strip()] for p in s.split("|") if p.strip() in CV_ACCENTS}
    return keys.pop() if len(keys) == 1 else None


# Map a free-form prompt-bank accent label to the unified label set.
def bank_accent(label):
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None
    a = str(label).strip().lower()
    a = ALIASES.get(a, a)
    if not a:
        return None
    a = SUBACCENT_PARENT.get(a, a)
    return BANK_TO_UNIFIED.get(a, a.replace(" ", "_"))


def condition(accent):
    return "ood" if accent in OOD else "heldout" if accent in HELDOUT else "seen"
