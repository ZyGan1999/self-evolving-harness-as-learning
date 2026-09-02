"""Persona: a named bundle of preference rules loaded from YAML."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from .episode import EpisodeRecord
from .history import SessionHistory
from .rules import Rule, RuleResult, build_rules


@dataclass
class Persona:
    name: str
    rules: list[Rule]
    description: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "Persona":
        cfg = yaml.safe_load(Path(path).read_text())
        return cls(name=cfg["name"], rules=build_rules(cfg["rules"]),
                   description=cfg.get("description", ""))

    def check(self, ep: EpisodeRecord, history: SessionHistory) -> list[RuleResult]:
        return [rule.check(ep, history) for rule in self.rules]

    def relevant_apps(self) -> set[str]:
        return {app for rule in self.rules for app in rule.apps}

    def trigger_apis(self) -> set[str]:
        """'{app}.{api}' strings that trigger at least one rule of this persona."""
        return {api for rule in self.rules for api in rule.trigger_apis()}

    def rules_triggered_by(self, required_apis: set[str]) -> list[str]:
        return [rule.name for rule in self.rules if rule.trigger_apis() & required_apis]

    def oracle_context(self, verbosity: int = 1) -> str:
        """Perfect natural-language description of all rules (oracle-c).
        verbosity 1/4/16 repeats-with-elaboration for the length-sweep control."""
        lines = ["My preferences for how you should do things:"]
        for i, rule in enumerate(self.rules, 1):
            lines.append(f"{i}. {rule.oracle_text}")
            for k in range(verbosity - 1):
                lines.append(f"   (Reminder {k + 1}: {rule.oracle_text})")
        return "\n".join(lines)
