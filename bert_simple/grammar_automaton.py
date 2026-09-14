"""A fixed, inspectable grammar automaton for masked-slot constraints.

The automaton does not learn grammar weights.  It matches local transitions
around a masked token and keeps a small stack for nested condition/parenthesis
contexts.  Word attributes are learned separately by ``WordAttributeLayer``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


ATTRIBUTES = (
    "PLACE",
    "TIME",
    "PERSON",
    "ORG",
    "TITLE",
    "NUMBER",
    "ACTION",
    "ENTITY",
    "FUNCTION",
    "PUNCT",
    "INTERROGATIVE",
    "CLAUSE",
    "UNKNOWN",
)

STRUCTURES = (
    "NAME",
    "SUBJECT",
    "LOCATION",
    "TEMPORAL",
    "PREDICATE",
    "OBJECT",
    "IDENTITY",
    "QUANTITY",
    "PASSIVE",
    "NEGATION",
    "CAUSATIVE",
    "CONDITION",
    "COORDINATION",
    "PARTICLE",
    "BOUNDARY",
    "OPEN",
)

DELIMITERS = frozenset(("，", "。", "！", "？", "；"))
CONDITION_OPENERS = frozenset(("如果", "若", "因为", "只要", "当", "即使", "虽然"))
CLAUSE_START_MARKERS = DELIMITERS | CONDITION_OPENERS | frozenset(
    ("：", "“", "‘", "（", "(", "且", "则", "而", "但")
)
SUBJECT_PREDICATES = frozenset(
    (
        "是",
        "为",
        "在",
        "有",
        "被",
        "派",
        "让",
        "使",
        "令",
        "命令",
        "开始",
        "已经",
        "正在",
        "还在",
        "率领",
        "前往",
        "到达",
        "住在",
        "出任",
        "任命",
        "成为",
        "担任",
        "攻打",
        "逃入",
        "逃到",
        "认为",
        "决定",
    )
)


@dataclass(frozen=True)
class GrammarRule:
    """One fixed transition pattern around a masked position."""

    name: str
    structure: str
    attributes: Tuple[str, ...]
    left: Tuple[str, ...] = ()
    right: Tuple[str, ...] = ()
    priority: int = 50


@dataclass(frozen=True)
class GrammarMatch:
    structure: str
    attributes: Tuple[str, ...]
    rule: str
    priority: int


@dataclass(frozen=True)
class GrammarState:
    """The state-machine result for one masked slot."""

    structure: str
    allowed_attributes: Tuple[str, ...]
    matches: Tuple[GrammarMatch, ...]
    fallback: bool
    stack: Tuple[str, ...] = ()


class FixedGrammarAutomaton:
    """Match fixed local grammar transitions and nested clause contexts."""

    def __init__(self, rules: Iterable[GrammarRule] | None = None):
        self.rules = tuple(rules or self.default_rules())

    @staticmethod
    def default_rules() -> Tuple[GrammarRule, ...]:
        person = ("PERSON", "ENTITY")
        place = ("PLACE", "ENTITY")
        subject = ("PERSON", "PLACE", "ORG", "TITLE", "ENTITY")
        # "被 + X" is ambiguous: X can be an agent (项羽) or a bare passive
        # predicate (被疏通).  Keep ACTION allowed until lexical context
        # disambiguates it.
        passive_agent = ("PERSON", "ORG", "TITLE", "ENTITY", "ACTION")
        broad_after_zai = (
            "PLACE",
            "TIME",
            "ACTION",
            "ENTITY",
            "FUNCTION",
            "INTERROGATIVE",
        )
        return (
            # Naming and identity slots.
            GrammarRule("name-叫", "NAME", person, ("叫",), priority=90),
            GrammarRule("name-名叫", "NAME", person, ("名叫",), priority=90),
            GrammarRule("name-名为", "NAME", person, ("名为",), priority=90),
            GrammarRule("name-姓名", "NAME", person, ("姓名",), priority=90),
            GrammarRule("name-姓名是", "NAME", person, ("姓名", "是"), priority=95),
            GrammarRule(
                "identity-是一位",
                "IDENTITY",
                ("TITLE", "PERSON", "ENTITY"),
                ("是", "一位"),
                priority=92,
            ),
            GrammarRule(
                "identity-是一名",
                "IDENTITY",
                ("TITLE", "PERSON", "ENTITY"),
                ("是", "一名"),
                priority=92,
            ),
            GrammarRule(
                "identity-担任",
                "IDENTITY",
                ("TITLE", "ORG", "ENTITY"),
                ("担任",),
                priority=88,
            ),
            GrammarRule(
                "identity-任命为",
                "IDENTITY",
                ("TITLE", "ORG", "ENTITY"),
                ("任命", "为"),
                priority=92,
            ),
            GrammarRule(
                "identity-任命-为",
                "IDENTITY",
                ("TITLE", "ORG", "ENTITY"),
                ("任命",),
                ("为",),
                94,
            ),
            GrammarRule(
                "identity-封为",
                "IDENTITY",
                ("TITLE", "ORG", "ENTITY"),
                ("封为",),
                priority=88,
            ),
            GrammarRule(
                "identity-封-为",
                "IDENTITY",
                ("TITLE", "ORG", "ENTITY"),
                ("封",),
                ("为",),
                94,
            ),
            GrammarRule(
                "identity-立-为",
                "IDENTITY",
                ("TITLE", "PERSON", "ORG", "ENTITY"),
                ("立",),
                ("为",),
                92,
            ),
            GrammarRule(
                "identity-被任命为",
                "IDENTITY",
                ("TITLE", "ORG", "ENTITY"),
                ("被", "任命", "为"),
                priority=94,
            ),
            GrammarRule(
                "identity-成为一名",
                "IDENTITY",
                ("TITLE", "PERSON", "ENTITY"),
                ("成为", "一名"),
                priority=92,
            ),
            # Location and temporal slots.
            GrammarRule("temporal-在时", "TEMPORAL", ("TIME",), ("在",), ("时",), 100),
            GrammarRule("temporal-在时候", "TEMPORAL", ("TIME",), ("在",), ("时候",), 100),
            GrammarRule("temporal-在期间", "TEMPORAL", ("TIME",), ("在",), ("期间",), 100),
            GrammarRule("temporal-于时", "TEMPORAL", ("TIME",), ("于",), ("时",), 100),
            GrammarRule("temporal-于时候", "TEMPORAL", ("TIME",), ("于",), ("时候",), 100),
            GrammarRule("temporal-于期间", "TEMPORAL", ("TIME",), ("于",), ("期间",), 100),
            GrammarRule("location-住在", "LOCATION", place, ("住在",), priority=88),
            GrammarRule("location-前往", "LOCATION", place, ("前往",), priority=88),
            GrammarRule("location-到达", "LOCATION", place, ("到达",), priority=88),
            GrammarRule("location-来到", "LOCATION", place, ("来到",), priority=88),
            GrammarRule("location-去", "LOCATION", place, ("去",), priority=82),
            GrammarRule("location-在", "LOCATION", broad_after_zai, ("在",), priority=62),
            GrammarRule(
                "location-到了",
                "LOCATION",
                place + ("FUNCTION", "INTERROGATIVE"),
                ("到了",),
                priority=62,
            ),
            # Predicate/object slots.
            GrammarRule("predicate-正在", "PREDICATE", ("ACTION",), ("正在",), priority=84),
            GrammarRule("predicate-已经", "PREDICATE", ("ACTION",), ("已经",), priority=84),
            GrammarRule("predicate-开始", "PREDICATE", ("ACTION",), ("开始",), priority=84),
            GrammarRule("predicate-还在", "PREDICATE", ("ACTION",), ("还在",), priority=84),
            GrammarRule("predicate-企图", "PREDICATE", ("ACTION",), ("企图",), priority=84),
            GrammarRule("predicate-决定", "PREDICATE", ("ACTION",), ("决定",), priority=84),
            GrammarRule("object-叫", "OBJECT", person, ("叫",), priority=72),
            GrammarRule("object-让", "OBJECT", person, ("让",), priority=76),
            GrammarRule("object-请", "OBJECT", person, ("请",), priority=76),
            GrammarRule("object-派", "OBJECT", person, ("派",), priority=76),
            GrammarRule("object-命令", "OBJECT", person, ("命令",), priority=76),
            # Passive transitions.
            GrammarRule("passive-agent-被", "PASSIVE", passive_agent, ("被",), priority=90),
            GrammarRule("passive-agent-遭", "PASSIVE", passive_agent, ("遭",), priority=84),
            # Negation transitions.
            GrammarRule("negation-不", "NEGATION", ("ACTION", "ENTITY"), ("不",), priority=80),
            GrammarRule("negation-未", "NEGATION", ("ACTION", "ENTITY"), ("未",), priority=80),
            GrammarRule("negation-没有", "NEGATION", ("ACTION", "ENTITY"), ("没有",), priority=80),
            GrammarRule("negation-无", "NEGATION", ("ACTION", "ENTITY"), ("无",), priority=80),
            GrammarRule("negation-未曾", "NEGATION", ("ACTION", "ENTITY"), ("未曾",), priority=80),
            GrammarRule("negation-莫", "NEGATION", ("ACTION", "ENTITY"), ("莫",), priority=80),
            # Causative/serial-verb transitions.
            GrammarRule("causative-派", "CAUSATIVE", person, ("派",), priority=86),
            GrammarRule("causative-让", "CAUSATIVE", person, ("让",), priority=86),
            GrammarRule("causative-使", "CAUSATIVE", person, ("使",), priority=86),
            GrammarRule("causative-令", "CAUSATIVE", person, ("令",), priority=86),
            GrammarRule("causative-命令", "CAUSATIVE", person, ("命令",), priority=86),
            GrammarRule("causative-叫-去", "CAUSATIVE", person, ("叫",), ("去",), 94),
            GrammarRule("causative-叫-来", "CAUSATIVE", person, ("叫",), ("来",), 94),
            # Quantity and coordination.
            GrammarRule("quantity-有", "QUANTITY", ("NUMBER",), ("有",), priority=82),
            GrammarRule("quantity-带了", "QUANTITY", ("NUMBER",), ("带了",), priority=82),
            GrammarRule("quantity-共有", "QUANTITY", ("NUMBER",), ("共有",), priority=82),
            GrammarRule("quantity-拥有", "QUANTITY", ("NUMBER",), ("拥有",), priority=82),
            GrammarRule("quantity-率领", "QUANTITY", ("NUMBER",), ("率领",), priority=72),
            GrammarRule("coordination-一起", "COORDINATION", ("FUNCTION",), (), ("*", "一起"), priority=78),
            GrammarRule("coordination-一同", "COORDINATION", ("FUNCTION",), (), ("*", "一同"), priority=78),
            GrammarRule("coordination-共同", "COORDINATION", ("FUNCTION",), (), ("*", "共同"), priority=78),
            GrammarRule("particle-来了", "PARTICLE", ("FUNCTION",), ("来了",), ("？",), priority=78),
            GrammarRule("particle-去了", "PARTICLE", ("FUNCTION",), ("去了",), ("？",), priority=78),
            GrammarRule("particle-还在", "PARTICLE", ("FUNCTION",), ("还", "在"), ("。",), priority=76),
            # Lower-priority question fallback.
            GrammarRule("open-在", "OPEN", ("INTERROGATIVE", "PLACE", "ACTION", "FUNCTION"), ("在",), ("？",), priority=35),
            GrammarRule("open-去", "OPEN", ("INTERROGATIVE", "PLACE", "ACTION", "FUNCTION"), ("去",), ("？",), priority=35),
            GrammarRule("open-叫", "OPEN", ("INTERROGATIVE", "PERSON", "ENTITY"), ("叫",), ("？",), priority=35),
        )

    @staticmethod
    def _pattern_matches(
        tokens: Sequence[str], start: int, pattern: Sequence[str], step: int
    ) -> bool:
        for offset, expected in enumerate(pattern):
            index = start + step * offset
            if index < 0 or index >= len(tokens):
                return False
            if expected != "*" and tokens[index] != expected:
                return False
        return True

    def _matches_rule(
        self, tokens: Sequence[str], mask_index: int, rule: GrammarRule
    ) -> bool:
        left_start = mask_index - len(rule.left)
        if not self._pattern_matches(tokens, left_start, rule.left, 1):
            return False
        return self._pattern_matches(tokens, mask_index + 1, rule.right, 1)

    @classmethod
    def _subject_match(
        cls, tokens: Sequence[str], mask_index: int
    ) -> GrammarMatch | None:
        # A masked subject may follow a topic phrase, so do not require the
        # token immediately before it to be punctuation.  Any recognized
        # predicate immediately to the right is enough to enter SUBJECT.
        if mask_index + 1 >= len(tokens):
            return None
        next_token = tokens[mask_index + 1]
        if next_token in SUBJECT_PREDICATES:
            return GrammarMatch(
                "SUBJECT",
                ("PERSON", "PLACE", "ORG", "TITLE", "ENTITY"),
                f"subject-before-{next_token}",
                66,
            )
        return None

    @classmethod
    def _passive_subject_match(
        cls, tokens: Sequence[str], mask_index: int
    ) -> GrammarMatch | None:
        if mask_index + 1 < len(tokens) and tokens[mask_index + 1] == "被":
            return GrammarMatch(
                "PASSIVE",
                ("PERSON", "PLACE", "ORG", "TITLE", "ENTITY"),
                "passive-subject-before-被",
                92,
            )
        return None

    @staticmethod
    def _scan_stack(tokens: Sequence[str], mask_index: int) -> Tuple[str, ...]:
        stack: List[str] = []
        for token in tokens[:mask_index]:
            if token in CONDITION_OPENERS:
                stack.append("CONDITION")
            elif token in {"（", "("}:
                stack.append("PAREN")
            elif token in {"）", ")"}:
                if stack and stack[-1] == "PAREN":
                    stack.pop()
            elif token == "，":
                if stack and stack[-1] == "CONDITION":
                    stack.pop()
            elif token in {"。", "！", "？", "；"}:
                stack.clear()
        return tuple(stack)

    @staticmethod
    def _condition_match(stack: Sequence[str]) -> GrammarMatch | None:
        if "CONDITION" in stack:
            return GrammarMatch("CONDITION", ("CLAUSE",), "condition-stack", 70)
        return None

    def analyze(self, tokens: Sequence[str], mask_index: int) -> GrammarState:
        if not 0 <= mask_index < len(tokens):
            raise IndexError("mask_index is outside tokens")
        stack = self._scan_stack(tokens, mask_index)
        matches: List[GrammarMatch] = []
        condition = self._condition_match(stack)
        if condition is not None:
            matches.append(condition)
        subject = self._subject_match(tokens, mask_index)
        if subject is not None:
            matches.append(subject)
        passive_subject = self._passive_subject_match(tokens, mask_index)
        if passive_subject is not None:
            matches.append(passive_subject)
        for rule in self.rules:
            if self._matches_rule(tokens, mask_index, rule):
                matches.append(
                    GrammarMatch(rule.structure, rule.attributes, rule.name, rule.priority)
                )
        if mask_index == len(tokens) - 1:
            matches.append(GrammarMatch("BOUNDARY", ("PUNCT",), "boundary-end", 100))
        if not matches:
            return GrammarState(
                "OPEN",
                tuple(a for a in ATTRIBUTES if a not in {"UNKNOWN", "PUNCT"}),
                (),
                True,
                stack,
            )
        best_priority = max(match.priority for match in matches)
        selected = tuple(match for match in matches if match.priority == best_priority)
        allowed = sorted({attribute for match in selected for attribute in match.attributes})
        return GrammarState(selected[0].structure, tuple(allowed), selected, False, stack)

    def analyze_text(self, text: str, mask_token: str = "[MASK]") -> GrammarState:
        tokens = text.split()
        if tokens.count(mask_token) != 1:
            raise ValueError("text must contain exactly one mask token")
        return self.analyze(tokens, tokens.index(mask_token))
