"""Units: spoken Spanish form -> canonical abbreviation ("miligramos" -> "mg").

Policy (part of the metric definition, not an implementation detail):

1. A unit is abbreviated **only when it follows a number**: "cada 8 horas" -> "cada 8 h", but
   "pasadas unas horas" is left alone. Without that condition the map would rewrite ordinary
   words that happen to be unit names.
2. The canonical form is the abbreviation, because it collapses the two ways a system may
   write a unit to a single string.
3. Singular and plural map to the same abbreviation ("miligramo" / "miligramos" -> "mg").
4. Multi-word units ("milímetros de mercurio", "por ciento") are matched before single-word
   ones (longest match first).
"""

from __future__ import annotations

#: spoken form (lowercase, no punctuation) -> canonical abbreviation
SPOKEN_TO_ABBREV: dict[str, str] = {
    # mass
    "miligramo": "mg",
    "miligramos": "mg",
    "microgramo": "mcg",
    "microgramos": "mcg",
    "gramo": "g",
    "gramos": "g",
    "kilogramo": "kg",
    "kilogramos": "kg",
    "kilo": "kg",
    "kilos": "kg",
    # volume
    "mililitro": "ml",
    "mililitros": "ml",
    "litro": "l",
    "litros": "l",
    # length
    "milímetro": "mm",
    "milímetros": "mm",
    "centímetro": "cm",
    "centímetros": "cm",
    # time
    "hora": "h",
    "horas": "h",
    "minuto": "min",
    "minutos": "min",
    "día": "d",
    "días": "d",
    "semana": "sem",
    "semanas": "sem",
    # clinical and compound (multi-word)
    "milímetros de mercurio": "mmhg",
    "milímetro de mercurio": "mmhg",
    "miliequivalente": "meq",
    "miliequivalentes": "meq",
    "unidad internacional": "ui",
    "unidades internacionales": "ui",
    "latidos por minuto": "lpm",
    "respiraciones por minuto": "rpm",
    "grados centígrados": "ºc",
    "por ciento": "%",
}

#: spelling variants of an abbreviation that map to the canonical one
ABBREV_ALIASES: dict[str, str] = {
    "µg": "mcg",
    "μg": "mcg",
    "ug": "mcg",
    "mgs": "mg",
    "gr": "g",
    "grs": "g",
    "cc": "ml",
    "hr": "h",
    "hrs": "h",
    "mins": "min",
    "°c": "ºc",
}

#: every word that can start a spoken unit (disambiguates "un gramo")
UNIT_START_WORDS: frozenset[str] = frozenset(k.split()[0] for k in SPOKEN_TO_ABBREV)

#: longest spoken unit, in tokens
MAX_UNIT_TOKENS: int = max(len(k.split()) for k in SPOKEN_TO_ABBREV)


def abbreviation(spoken: str) -> str | None:
    """Canonical abbreviation of a spoken unit, or None if it is not a known unit."""
    return SPOKEN_TO_ABBREV.get(spoken)
