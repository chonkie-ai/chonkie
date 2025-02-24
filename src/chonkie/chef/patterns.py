"""Common text patterns and abbreviations used across chefs."""

from dataclasses import dataclass

# Titles and honorifics
TITLES = {
    "Mr.",
    "Mrs.",
    "Ms.",
    "Dr.",
    "Prof.",
    "Sr.",
    "Jr.",
    "Rev.",
    "Hon.",
}

# Academic and professional
ACADEMIC = {
    "Ph.D.",
    "M.D.",
    "B.A.",
    "M.A.",
    "B.Sc.",
    "M.Sc.",
    "D.Phil.",
    "LL.B.",
    "LL.M.",
}

# Latin abbreviations
LATIN = {
    "etc.",
    "e.g.",
    "i.e.",
    "viz.",
    "vs.",
    "al.",
    "et al.",
    "cf.",
}

# Military and government
MILITARY = {
    "Gen.",
    "Col.",
    "Lt.",
    "Sgt.",
    "Capt.",
    "Maj.",
    "Adm.",
    "Gov.",
    "Sen.",
    "Rep.",
}

# Common measurements
MEASUREMENTS = {
    "cm.",
    "mm.",
    "km.",
    "kg.",
    "lb.",
    "ft.",
    "in.",
    "hr.",
    "min.",
    "sec.",
}

# Business and organization
BUSINESS = {
    "Inc.",
    "Ltd.",
    "Corp.",
    "Co.",
    "LLC.",
    "dept.",
    "div.",
    "est.",
    "avg.",
    "approx.",
}

# Temporal abbreviations
TEMPORAL = {
    "Jan.",
    "Feb.",
    "Mar.",
    "Apr.",
    "Jun.",
    "Jul.",
    "Aug.",
    "Sep.",
    "Sept.",
    "Oct.",
    "Nov.",
    "Dec.",
    "Mon.",
    "Tue.",
    "Wed.",
    "Thu.",
    "Fri.",
    "Sat.",
    "Sun.",
}

# Geographical abbreviations
GEOGRAPHICAL = {
    "U.S.",
    "U.S.A.",
    "U.K.",
    "E.U.",
    "Ave.",
    "Blvd.",
    "Rd.",
    "St.",
    "Mt.",
}

# Combine all abbreviations
ABBREVIATIONS = (
    TITLES
    | ACADEMIC
    | LATIN
    | MILITARY
    | MEASUREMENTS
    | BUSINESS
    | TEMPORAL
    | GEOGRAPHICAL
)


@dataclass
class UnicodeReplacements:

    """Common Unicode characters used for replacements."""

    DOT_LEADER = "․"  # U+2024 ONE DOT LEADER
    ELLIPSIS = "…"  # U+2026 HORIZONTAL ELLIPSIS
