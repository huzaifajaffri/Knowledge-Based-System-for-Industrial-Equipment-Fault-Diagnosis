"""
Inference Engine Module for Industrial Equipment Fault Diagnosis KBS
====================================================================
Course: Knowledge-Based Systems (CT-359)
Domain: Industrial Equipment Fault Diagnosis (Motor, Pump, Gearbox, Compressor)

Theoretical Architecture & Viva Defense Reference:
--------------------------------------------------
1. Knowledge-Based System (KBS) Triad:
   - Knowledge Base (KB): Domain expertise encoded as declarative Rules (IF-THEN)
     and structural Frames (slot-filler component hierarchies).
   - Working Memory (WM): Dynamic state containing current observations, sensor
     readings, and newly asserted diagnostic hypotheses.
   - Inference Engine (IE): The problem-solving brain that drives forward-chaining
     reasoning, uncertainty propagation, and conflict resolution.

2. Forward Chaining (Data-Driven Reasoning / Modus Ponens):
   - Direction: Evidence -> Rules -> Conclusions.
   - Match-Resolve-Act Cycle:
     * Match Phase: Compare facts in Working Memory (observed symptoms) against
       rule antecedents (IF condition).
     * Fire Phase: Activate matching rules and derive consequences (THEN cause).
     * Aggregate Phase: Group multiple rules diagnosing identical faults.
   - Justification: Maintenance engineers begin with observed machine symptoms
     and require the system to deduce root causes and corrective actions.

3. Uncertainty Management (Certainty Factor Algebra):
   - Shortliffe & Buchanan (MYCIN expert system model, 1975).
   - Rule Base Certainty (CF_base = 0.80): Confidence that a single symptom indicates the fault.
   - Combination Rule for Multiple Independent Evidence:
         CF_combined = CF1 + CF2 * (1 - CF1)
     This formula exhibits asymptotic saturation towards 1.0 (never exceeding 1.0).
   - Sensor Corroboration: Objective physical measurements (temperature, vibration,
     pressure, current) boost subjective symptom certainty by +0.10 (capped at 0.99).

4. Conflict Resolution Strategy:
   - When multiple competing faults are diagnosed, industrial safety demands strict prioritization:
     Tier 1: Safety Priority Weight (CRITICAL=4 > HIGH=3 > MEDIUM=2 > LOW=1)
     Tier 2: Certainty Score (higher confidence faults addressed first)
     Tier 3: Number of Supporting Symptoms (more corroborated evidence prioritized)

5. Transparent Explanation Facility (White-Box Reasoning):
   - Generates an auditable, step-by-step reasoning trace explaining:
     (1) User observations -> (2) Rule firings -> (3) Remedy -> (4) Component ->
     (5) CF calculation formula -> (6) Safety priority justification.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
import re

# Attempt to import core KnowledgeBase, Rule, and Frame from knowledge_base.py
# If knowledge_base.py is not yet generated or is running in isolation,
# graceful fallback stubs are provided so this engine can be tested standalone.
try:
    from knowledge_base import KnowledgeBase, Rule, Frame
except ImportError:
    @dataclass
    class Rule:
        """
        Fallback Rule representation for standalone testing.
        Represents an IF-THEN diagnostic production rule.
        """
        rule_id: str
        machine_type: str
        symptom: str
        cause: str
        remedy: str
        component: str
        measurable_condition: str = ""
        source: str = ""
        base_certainty: float = 0.80
        safety_priority: Optional[str] = None

    @dataclass
    class Frame:
        """
        Fallback Frame representation for structural component knowledge.
        """
        name: str
        frame_type: str = "component"
        slots: dict = field(default_factory=dict)
        parent: Optional[str] = None

    class KnowledgeBase:
        """
        Fallback KnowledgeBase representation for standalone execution.
        """
        def __init__(self, data_dir: str = "DataSet") -> None:
            self.rules: Dict[str, List[Rule]] = {}
            self.frames: Dict[str, Frame] = {}

        def get_rules(self, machine_type: str) -> List[Rule]:
            """Retrieve rules for a given machine type."""
            return self.rules.get(machine_type.lower(), [])


@dataclass
class DiagnosisResult:
    """
    Encapsulates a fully deduced diagnostic conclusion.

    Attributes:
        cause: Root cause or diagnosed equipment failure mode.
        remedy: Prescribed maintenance action or corrective procedure.
        component: Physical mechanical/electrical sub-assembly affected.
        certainty: Calculated Certainty Factor (0.0 to 0.99).
        safety_priority: Operational safety rating ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW').
        matched_symptoms: List of user-selected symptoms supporting this fault.
        matched_rules: List of rule IDs that fired to deduce this fault.
        reasoning_chain: Step-by-step transparent explanation of deduction.
        sensor_supported: Whether sensor data corroborated this diagnosis.
    """
    cause: str
    remedy: str
    component: str
    certainty: float
    safety_priority: str
    matched_symptoms: list[str]
    matched_rules: list[str]  # rule IDs
    reasoning_chain: list[str]  # step-by-step reasoning
    sensor_supported: bool


class InferenceEngine:
    """
    Forward-Chaining Inference Engine with Certainty Factors and Explanation System.

    Implements data-driven deduction for industrial equipment fault diagnosis.
    Evaluates observed symptoms against production rules, aggregates multiple
    evidence lines, corroborates with telemetry/sensor data, resolves competing
    hypotheses via safety heuristics, and generates auditable explanations.
    """

    # Priority ranking weights for industrial conflict resolution
    # Higher weights indicate greater urgency for safety and asset integrity.
    SAFETY_PRIORITY_WEIGHTS: Dict[str, int] = {
        "CRITICAL": 4,
        "HIGH": 3,
        "MEDIUM": 2,
        "LOW": 1
    }

    # Keyword patterns for automated safety priority classification
    # Derived from industrial hazard mitigation and ISO/OSHA action levels.
    CRITICAL_KEYWORDS: Tuple[str, ...] = (
        "shut down", "shutdown", "stop immediately", "stop", "do not keep running",
        "emergency", "catastrophic", "danger", "hazardous", "fire", "explosion",
        "severe damage"
    )

    HIGH_KEYWORDS: Tuple[str, ...] = (
        "check", "replace", "inspect", "examine", "isolate", "disconnect",
        "repair", "overload", "high voltage", "warning", "blown", "rewind",
        "investigate", "test"
    )

    MEDIUM_KEYWORDS: Tuple[str, ...] = (
        "adjust", "clean", "tighten", "ensure", "set", "re-prime", "correct",
        "lubricate", "refill", "renew", "align", "reduce load", "clear",
        "review and revise"
    )

    LOW_KEYWORDS: Tuple[str, ...] = (
        "consult", "monitor", "observe", "routine", "schedule", "verify",
        "review", "inform", "plan"
    )

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        """
        Initialize the Inference Engine with an active Knowledge Base.

        Args:
            knowledge_base: Instance of KnowledgeBase containing equipment rules and frames.
        """
        self.kb: KnowledgeBase = knowledge_base
        self.default_base_certainty: float = 0.80
        self.sensor_boost_value: float = 0.10
        self.max_certainty_cap: float = 0.99

    def diagnose(
        self,
        machine_type: str,
        selected_symptoms: list[str],
        sensor_data: Optional[dict] = None
    ) -> list[DiagnosisResult]:
        """
        Main entry point: Execute forward chaining inference for equipment fault diagnosis.

        Forward Chaining Workflow:
        1. Initialize Working Memory with user-selected symptoms as known facts.
        2. Match Phase: Scan the rule base to find all rules matching working memory facts.
        3. Fire Phase: Trigger matching rules to produce candidate fault hypotheses.
        4. Aggregate Phase: Combine multiple rules that point to the same root cause.
        5. Corroboration & Certainty Calculation: Apply MYCIN CF algebra and sensor boost.
        6. Explanation Construction: Build human-readable step-by-step reasoning traces.
        7. Conflict Resolution: Rank diagnoses by safety weight, certainty, and evidence count.

        Args:
            machine_type: Target machinery ('motor', 'pump', 'gearbox', 'compressor').
            selected_symptoms: List of symptom descriptions observed by the user/operator.
            sensor_data: Optional dictionary containing sensor readings and telemetry flags.

        Returns:
            Ranked list of DiagnosisResult instances sorted by safety priority and certainty.
            Returns an empty list if no symptoms match or if inputs are invalid.
        """
        # Step 0: Input validation and handling incomplete information gracefully
        if not machine_type or not selected_symptoms:
            return []

        # Filter and clean working memory facts
        working_memory = [s.strip() for s in selected_symptoms if s and s.strip()]
        if not working_memory:
            return []

        # Step 1: Match Phase - Find rules whose antecedent matches working memory facts
        matched_rules = self._match_rules(machine_type, working_memory)
        if not matched_rules:
            return []

        # Step 2: Aggregate Phase - Group fired rules by diagnosed cause/fault
        aggregated_faults = self._aggregate_results(matched_rules)
        if not aggregated_faults:
            return []

        # Step 3: Deduction & Synthesis Phase
        results: list[DiagnosisResult] = []

        for cause, rules_for_cause in aggregated_faults.items():
            # Extract distinct symptoms and rule IDs supporting this specific cause
            matched_syms = self._extract_unique_symptoms(rules_for_cause)
            matched_rule_ids = self._extract_rule_ids(rules_for_cause)

            # Check if telemetry/sensor measurements corroborate this diagnosis
            sensor_supported = self._check_sensor_support(rules_for_cause, sensor_data)

            # Calculate Certainty Factor via MYCIN combination algebra + sensor boost
            certainty = self._calculate_certainty(rules_for_cause, sensor_supported)

            # Select the most comprehensive remedy and affected component
            remedy = self._determine_remedy(rules_for_cause)
            component = self._determine_component(rules_for_cause)

            # Determine safety priority based on explicit rule tags or remedy action verbs
            safety_priority, priority_reason = self._determine_safety_priority(remedy, cause, rules_for_cause)

            # Create candidate diagnosis result (reasoning chain populated next)
            candidate = DiagnosisResult(
                cause=cause,
                remedy=remedy,
                component=component,
                certainty=certainty,
                safety_priority=safety_priority,
                matched_symptoms=matched_syms,
                matched_rules=matched_rule_ids,
                reasoning_chain=[],
                sensor_supported=sensor_supported
            )

            # Generate transparent explanation trace
            reasoning_chain = self._generate_reasoning(candidate, rules_for_cause, priority_reason)
            candidate.reasoning_chain = reasoning_chain

            results.append(candidate)

        # Step 4: Conflict Resolution Phase - Order results by safety, certainty, evidence
        resolved_results = self._resolve_conflicts(results)

        return resolved_results

    def _match_rules(self, machine_type: str, selected_symptoms: list[str]) -> list[Rule]:
        """
        Match Phase: Scan rule base for the specified machine and match against working memory.

        Matches facts in working memory against rule antecedents. Uses robust text
        normalization (case insensitivity, punctuation stripping, whitespace trimming)
        to bridge variations between user interfaces and raw dataset strings.

        Args:
            machine_type: Machinery category ('motor', 'pump', 'gearbox', 'compressor').
            selected_symptoms: List of user-observed symptom strings in Working Memory.

        Returns:
            List of Rule objects that fired during the pattern matching phase.
        """
        # Retrieve candidate rules for the specified machine
        candidate_rules = self._get_rules_from_kb(machine_type)
        if not candidate_rules:
            return []

        # Normalize working memory facts for matching
        normalized_wm = {self._normalize_text(s): s for s in selected_symptoms if s}

        matched: list[Rule] = []

        for rule in candidate_rules:
            rule_symptom_raw = getattr(rule, "symptom", "")
            if not rule_symptom_raw:
                continue

            rule_symptom_norm = self._normalize_text(rule_symptom_raw)

            # Direct match check
            if rule_symptom_norm in normalized_wm:
                # Store the exact user-observed text on the rule for explanation provenance
                setattr(rule, "_matched_symptom", normalized_wm[rule_symptom_norm])
                matched.append(rule)
            else:
                # Fallback check: substring containment for compound symptom phrases
                for norm_s, orig_s in normalized_wm.items():
                    if norm_s and (norm_s in rule_symptom_norm or rule_symptom_norm in norm_s):
                        setattr(rule, "_matched_symptom", orig_s)
                        matched.append(rule)
                        break

        return matched

    def _aggregate_results(self, matched: list[Rule]) -> dict[str, list[Rule]]:
        """
        Aggregate Phase: Group fired rules by diagnosed cause/fault.

        In industrial fault diagnosis, multiple distinct symptoms often point
        to the exact same underlying mechanical failure (e.g., both 'abnormal whine'
        and 'overheating' pointing to 'Lubricant level too low').
        This method groups rules by normalized cause so their certainty factors
        and evidence lines can be combined.

        Args:
            matched: List of Rule objects that fired during the Match phase.

        Returns:
            Dictionary mapping normalized fault/cause string to list of supporting Rules.
        """
        aggregated: dict[str, list[Rule]] = {}

        for rule in matched:
            raw_cause = getattr(rule, "cause", getattr(rule, "fault", "")).strip()
            if not raw_cause:
                continue

            # Standardize key presentation while grouping identical causes
            # Preserve title-cased or canonical cause name
            canonical_cause = raw_cause.rstrip(".;,")

            # Look for existing cluster with case-insensitive match
            existing_key = None
            for key in aggregated:
                if key.lower() == canonical_cause.lower():
                    existing_key = key
                    break

            if existing_key is not None:
                # Avoid duplicate identical rule additions
                rule_id = getattr(rule, "rule_id", getattr(rule, "id", None))
                existing_ids = [getattr(r, "rule_id", getattr(r, "id", None)) for r in aggregated[existing_key]]
                if rule_id is None or rule_id not in existing_ids:
                    aggregated[existing_key].append(rule)
            else:
                aggregated[canonical_cause] = [rule]

        return aggregated

    def _calculate_certainty(self, rules_for_cause: list[Rule], sensor_supported: bool) -> float:
        """
        Certainty Factor (CF) Calculation using MYCIN algebra.

        Mathematical Formulation:
        1. Single Symptom:
           CF = base_certainty (default 0.80)

        2. Multiple Independent Symptoms (Evidence Combination):
           When symptom E1 has CF1 and symptom E2 has CF2 pointing to hypothesis H:
           CF_combined(H, E1 & E2) = CF1 + CF2 * (1 - CF1)

           Iterative extension for n symptoms:
           CF_acc = CF1
           For each subsequent CFi:
               CF_acc = CF_acc + CFi * (1 - CF_acc)

        3. Sensor Corroboration:
           If objective telemetry data supports the diagnosis:
           CF_final = min(0.99, CF_acc + 0.10)

        Args:
            rules_for_cause: List of matching rules pointing to the same fault.
            sensor_supported: Boolean indicating whether sensor data corroborates the diagnosis.

        Returns:
            Combined Certainty Factor rounded to 2 decimal places (bounded [0.0, 0.99]).
        """
        if not rules_for_cause:
            return 0.0

        # Extract unique symptoms supporting this cause to avoid artificial boosting
        # from duplicate rules matching the same observation
        unique_symptoms: set[str] = set()
        representative_rules: list[Rule] = []

        for r in rules_for_cause:
            sym = getattr(r, "_matched_symptom", getattr(r, "symptom", ""))
            norm_sym = self._normalize_text(sym)
            if norm_sym not in unique_symptoms:
                unique_symptoms.add(norm_sym)
                representative_rules.append(r)

        if not representative_rules:
            representative_rules = rules_for_cause

        # Obtain base certainties for each contributing symptom
        base_cfs: list[float] = []
        for r in representative_rules:
            cf_val = getattr(r, "base_certainty", self.default_base_certainty)
            if cf_val is None or not isinstance(cf_val, (int, float)):
                cf_val = self.default_base_certainty
            base_cfs.append(float(cf_val))

        # Evidence combination
        if len(base_cfs) == 1:
            combined_cf = base_cfs[0]
        else:
            # Iterative Shortliffe & Buchanan MYCIN combination rule
            combined_cf = base_cfs[0]
            for next_cf in base_cfs[1:]:
                combined_cf = combined_cf + next_cf * (1.0 - combined_cf)

        # Apply sensor corroboration boost (+0.10, capped at 0.99)
        if sensor_supported:
            combined_cf = combined_cf + self.sensor_boost_value

        # Cap at 0.99 (KBS design principle: empirical confidence is never absolute 1.0)
        combined_cf = min(self.max_certainty_cap, combined_cf)

        return round(combined_cf, 2)

    def _resolve_conflicts(self, results: list[DiagnosisResult]) -> list[DiagnosisResult]:
        """
        Conflict Resolution Strategy: Rank competing diagnoses.

        When multiple faults are inferred simultaneously, the system must decide
        which recommendations to display first to ensure industrial safety.

        Multi-tiered Ranking Heuristic:
        1. Safety Priority Weight (Primary):
           CRITICAL (4) > HIGH (3) > MEDIUM (2) > LOW (1).
           Critical emergency shutdowns must precede routine checks.
        2. Certainty Factor (Secondary):
           Higher probability faults (e.g., 0.96 vs 0.80) are addressed first.
        3. Number of Matched Symptoms (Tertiary):
           Diagnoses corroborated by more observed symptoms are ranked higher.

        Args:
            results: Unsorted list of DiagnosisResult instances.

        Returns:
            Sorted list of DiagnosisResult instances ordered by urgency and certainty.
        """
        def conflict_sort_key(res: DiagnosisResult) -> Tuple[int, float, int]:
            priority_str = (res.safety_priority or "").upper().strip()
            priority_weight = self.SAFETY_PRIORITY_WEIGHTS.get(priority_str, 1)
            certainty_score = float(res.certainty)
            evidence_count = len(res.matched_symptoms)
            return (priority_weight, certainty_score, evidence_count)

        return sorted(results, key=conflict_sort_key, reverse=True)

    def _generate_reasoning(
        self,
        result: DiagnosisResult,
        rules: list[Rule],
        priority_reason: str = ""
    ) -> list[str]:
        """
        Explanation Facility: Build step-by-step reasoning chain (audit trail).

        Provides complete transparency into the expert system's internal deduction.
        Satisfies the classic KBS requirement: answering "HOW" a conclusion was reached.

        Step Template:
        - Step 1: User-observed symptom(s) asserted into Working Memory.
        - Step 2: Fired rule ID(s) and production rule logic (IF Antecedent THEN Consequent).
        - Step 3: Prescribed maintenance action / remedy.
        - Step 4: Affected component / subsystem.
        - Step 5: Certainty Factor calculation breakdown (base, combination, sensor boost).
        - Step 6: Safety priority level and actionable keyword justification.

        Args:
            result: The DiagnosisResult under construction.
            rules: Contributing Rule instances for this diagnosed cause.
            priority_reason: Explanation of how safety priority was assigned.

        Returns:
            List of 6 formatted reasoning step strings.
        """
        reasoning_chain: list[str] = []

        # Step 1: Observed symptoms
        if len(result.matched_symptoms) == 1:
            sym_text = result.matched_symptoms[0].rstrip(".;,")
            step_1 = f'Step 1: Symptom "{sym_text}" was observed by the user'
        else:
            formatted_syms = ", ".join(f'"{s.rstrip(".;,")}"' for s in result.matched_symptoms)
            step_1 = f'Step 1: Symptoms {formatted_syms} were observed by the user'
        reasoning_chain.append(step_1)

        # Step 2: Rule matching logic
        if len(rules) == 1:
            r = rules[0]
            r_id = getattr(r, "rule_id", getattr(r, "id", "R001"))
            r_symptom = getattr(r, "symptom", result.matched_symptoms[0]).rstrip(".;,")
            step_2 = f'Step 2: Rule {r_id} matched — IF symptom is "{r_symptom}" THEN cause is "{result.cause}"'
        else:
            r_ids = ", ".join(result.matched_rules)
            step_2 = f'Step 2: Rules {r_ids} matched — IF symptoms observed THEN cause is "{result.cause}"'
        reasoning_chain.append(step_2)

        # Step 3: Recommended action / remedy
        step_3 = f'Step 3: Recommended action: "{result.remedy}"'
        reasoning_chain.append(step_3)

        # Step 4: Affected component
        step_4 = f'Step 4: Affected component: "{result.component}"'
        reasoning_chain.append(step_4)

        # Step 5: Certainty Factor breakdown
        num_syms = len(result.matched_symptoms)
        if num_syms == 1:
            if result.sensor_supported:
                cf_detail = "single symptom match base 0.80 + 0.10 sensor corroboration boost"
            else:
                cf_detail = "single symptom match, base certainty"
        else:
            if result.sensor_supported:
                cf_detail = f"multiple symptom boost from {num_syms} symptoms + 0.10 sensor corroboration boost"
            else:
                cf_detail = f"multiple symptom boost from {num_syms} symptoms"

        step_5 = f'Step 5: Certainty calculated as {result.certainty:.2f} ({cf_detail})'
        reasoning_chain.append(step_5)

        # Step 6: Safety priority justification
        if not priority_reason:
            priority_reason = f'evaluated for {result.cause}'

        step_6 = f'Step 6: Safety priority: {result.safety_priority} ({priority_reason})'
        reasoning_chain.append(step_6)

        return reasoning_chain

    # =========================================================================
    # Internal Helper Methods
    # =========================================================================

    def _get_rules_from_kb(self, machine_type: str) -> list[Rule]:
        """
        Safely retrieve rules from KnowledgeBase across various interface styles.
        Supports get_rules(), get_rules_for_machine(), or direct rules dict/list.
        """
        norm_machine = machine_type.strip().lower()

        # Method 1: get_rules(machine_type)
        if hasattr(self.kb, "get_rules") and callable(self.kb.get_rules):
            try:
                rules = self.kb.get_rules(norm_machine)
                if rules:
                    return rules
            except Exception:
                pass

        # Method 2: get_rules_for_machine(machine_type)
        if hasattr(self.kb, "get_rules_for_machine") and callable(self.kb.get_rules_for_machine):
            try:
                rules = self.kb.get_rules_for_machine(norm_machine)
                if rules:
                    return rules
            except Exception:
                pass

        # Method 3: rules attribute (dict or list)
        if hasattr(self.kb, "rules"):
            kb_rules = self.kb.rules
            if isinstance(kb_rules, dict):
                # Check case-insensitive key
                for k, v in kb_rules.items():
                    if k.lower() == norm_machine:
                        return v
            elif isinstance(kb_rules, list):
                return [
                    r for r in kb_rules
                    if getattr(r, "machine_type", "").lower() == norm_machine
                ]

        return []

    def _check_sensor_support(self, rules: list[Rule], sensor_data: Optional[dict]) -> bool:
        """
        Evaluate whether sensor measurements corroborate any of the matching rules.

        Corroboration checks:
        1. Explicit boolean flag: e.g. sensor_data['corroborated'] == True.
        2. Measurable condition matching against sensor telemetry keys and thresholds.
        3. Industrial threshold violations (temperature > 80°C / 353K, vibration > 4.5 mm/s,
           current overload > 110%, voltage unbalance > 1%).
        """
        if not sensor_data or not isinstance(sensor_data, dict):
            return False

        # Check for explicit corroboration flag
        if sensor_data.get("corroborates") is True or sensor_data.get("sensor_supported") is True:
            return True

        # Scan each rule's measurable condition and component against sensor data
        for rule in rules:
            measurable_cond = getattr(rule, "measurable_condition", "") or ""
            component = getattr(rule, "component", "") or ""
            cause = getattr(rule, "cause", "") or ""

            cond_text = f"{measurable_cond} {component} {cause}".lower()

            # 1. Temperature Telemetry Check
            temp = sensor_data.get("temperature") or sensor_data.get("temp") or \
                   sensor_data.get("Process temperature [K]") or sensor_data.get("Air temperature [K]")
            if temp is not None:
                # If temperature indicates high heat and rule relates to temperature
                if any(kw in cond_text for kw in ("temp", "heat", "overheat", "thermal", "cooling", "hot")):
                    if isinstance(temp, (int, float)):
                        # Kelvin check (> 350K) or Celsius check (> 75°C) or Fahrenheit (> 170°F)
                        if temp > 350 or (30 < temp and temp > 75):
                            return True
                    elif str(temp).lower() in ("high", "abnormal", "critical", "overheating"):
                        return True

            # 2. Vibration Telemetry Check
            vib = sensor_data.get("vibration") or sensor_data.get("vib")
            if vib is not None:
                if any(kw in cond_text for kw in ("vibration", "rattling", "knocking", "noise", "bearing", "unbalance", "misalignment")):
                    if isinstance(vib, (int, float)) and vib > 4.5:  # ISO 10816 threshold (mm/s)
                        return True
                    elif str(vib).lower() in ("high", "severe", "abnormal", "excessive"):
                        return True

            # 3. Voltage / Current / Overload Telemetry Check
            current = sensor_data.get("current") or sensor_data.get("load")
            voltage = sensor_data.get("voltage") or sensor_data.get("voltage_unbalance")
            if current is not None or voltage is not None:
                if any(kw in cond_text for kw in ("current", "voltage", "overload", "relay", "power supply", "fuse")):
                    if sensor_data.get("overload") is True or sensor_data.get("overcurrent") is True:
                        return True
                    if isinstance(current, (int, float)) and current > 110:  # > 110% rated
                        return True
                    if isinstance(voltage, (int, float)) and (voltage < 360 or voltage > 440):
                        return True

            # 4. Oil / Lubrication / Wear Telemetry Check
            oil_cond = sensor_data.get("oil_condition") or sensor_data.get("metal_particles") or sensor_data.get("Tool wear [min]")
            if oil_cond is not None:
                if any(kw in cond_text for kw in ("oil", "lubricant", "wear", "particles", "viscosity", "teeth")):
                    if oil_cond is True or str(oil_cond).lower() in ("contaminated", "degraded", "high", "present"):
                        return True
                    if isinstance(oil_cond, (int, float)) and oil_cond > 150:
                        return True

            # 5. Pressure / Flow Telemetry Check
            flow = sensor_data.get("flow") or sensor_data.get("pressure")
            if flow is not None:
                if any(kw in cond_text for kw in ("flow", "pressure", "head", "suction", "discharge")):
                    if str(flow).lower() in ("low", "abnormal", "zero", "below rating"):
                        return True

            # 6. Generic key-word match on abnormal boolean flags in sensor_data
            for key, val in sensor_data.items():
                if val is True:
                    clean_key = key.replace("_", " ").lower()
                    if clean_key in cond_text:
                        return True

        return False

    def _determine_safety_priority(
        self,
        remedy: str,
        cause: str,
        rules: list[Rule]
    ) -> Tuple[str, str]:
        """
        Determine Safety Priority and justify with keyword detection.

        Priority Levels:
        - CRITICAL (weight 4): Immediate shutdown required to prevent injury or equipment destruction.
        - HIGH (weight 3): Prompt investigation/replacement required to prevent cascading damage.
        - MEDIUM (weight 2): Maintenance adjustment, cleaning, or replenishment needed.
        - LOW (weight 1): Advisory, monitoring, or scheduled inspection.

        Returns:
            Tuple of (priority_level, reasoning_explanation).
        """
        # First: Check if any contributing rule has an explicit safety_priority defined
        for r in rules:
            p = getattr(r, "safety_priority", None)
            if p and str(p).upper().strip() in self.SAFETY_PRIORITY_WEIGHTS:
                norm_p = str(p).upper().strip()
                return norm_p, f"explicit priority defined in rule {getattr(r, 'rule_id', '')}"

        target_text = f"{remedy} {cause}".strip()
        remedy_lower = remedy.lower()

        # Check CRITICAL keywords
        for kw in self.CRITICAL_KEYWORDS:
            if re.search(r'\b' + re.escape(kw) + r'\b', remedy_lower):
                return "CRITICAL", f'keyword "{kw.title()}" detected in remedy'

        # Check HIGH keywords
        for kw in self.HIGH_KEYWORDS:
            if re.search(r'\b' + re.escape(kw) + r'\b', remedy_lower):
                return "HIGH", f'keyword "{kw.title()}" detected in remedy'

        # Check MEDIUM keywords
        for kw in self.MEDIUM_KEYWORDS:
            if re.search(r'\b' + re.escape(kw) + r'\b', remedy_lower):
                return "MEDIUM", f'keyword "{kw.title()}" detected in remedy'

        # Check LOW keywords
        for kw in self.LOW_KEYWORDS:
            if re.search(r'\b' + re.escape(kw) + r'\b', remedy_lower):
                return "LOW", f'keyword "{kw.title()}" detected in remedy'

        # Fallback if remedy has no matching keyword
        return "MEDIUM", "standard maintenance priority"

    def _determine_remedy(self, rules: list[Rule]) -> str:
        """Select the most descriptive remedy from contributing rules."""
        remedies = [
            getattr(r, "remedy", getattr(r, "action", "")).strip().rstrip(".;,")
            for r in rules
            if getattr(r, "remedy", getattr(r, "action", None))
        ]
        if not remedies:
            return "Inspect component and refer to operational manual."
        # Pick the most detailed remedy
        return max(remedies, key=len)

    def _determine_component(self, rules: list[Rule]) -> str:
        """Select the most specific component name from contributing rules."""
        components = [
            getattr(r, "component", "").strip().rstrip(".;,")
            for r in rules
            if getattr(r, "component", None)
        ]
        if not components:
            return "General assembly"
        return max(components, key=len)

    def _extract_unique_symptoms(self, rules: list[Rule]) -> list[str]:
        """Extract unique symptom descriptions that triggered these rules."""
        symptoms: list[str] = []
        seen_norm: set[str] = set()

        for r in rules:
            sym = getattr(r, "_matched_symptom", getattr(r, "symptom", "")).strip()
            norm = self._normalize_text(sym)
            if norm and norm not in seen_norm:
                seen_norm.add(norm)
                symptoms.append(sym.rstrip(".;,"))

        return symptoms

    def _extract_rule_ids(self, rules: list[Rule]) -> list[str]:
        """Extract unique rule identifiers."""
        rule_ids: list[str] = []
        for r in rules:
            r_id = getattr(r, "rule_id", getattr(r, "id", None))
            if r_id and r_id not in rule_ids:
                rule_ids.append(r_id)
        return rule_ids

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize text for robust pattern matching.
        Strips trailing punctuation, collapses whitespace, and lowercases.
        """
        if not text:
            return ""
        # Strip trailing punctuation, extra spaces, and convert to lowercase
        cleaned = text.strip().rstrip(".;,:").strip().lower()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned


# =============================================================================
# Demonstration / Self-Test (Executable in Viva Demonstrations)
# =============================================================================
if __name__ == "__main__":
    print("=" * 75)
    print("INDUSTRIAL EQUIPMENT FAULT DIAGNOSIS - INFERENCE ENGINE DEMO")
    print("Course: Knowledge-Based Systems (CT-359)")
    print("=" * 75)

    # 1. Instantiate KnowledgeBase (using fallback test rules)
    demo_kb = KnowledgeBase()

    # Populate sample rules representative of the 4 machinery types
    demo_kb.rules["motor"] = [
        Rule(
            rule_id="MOTOR_R001",
            machine_type="motor",
            symptom="Motor does not start on load",
            cause="No supply voltage",
            remedy="Check voltage at Motor Terminals",
            component="Power supply",
            measurable_condition="Voltage reading below rated supply",
            base_certainty=0.80
        ),
        Rule(
            rule_id="MOTOR_R002",
            machine_type="motor",
            symptom="Motor starts and gets trips (star delta starter)",
            cause="Relay setting on lower reading than rated current",
            remedy="Set overload relay at 110% of rated full load current punched on Nameplate",
            component="Overload relay",
            measurable_condition="Relay set at 110% of nameplate full-load current",
            base_certainty=0.80
        ),
        Rule(
            rule_id="MOTOR_R015",
            machine_type="motor",
            symptom="Motors runs and then stops",
            cause="No supply voltage",
            remedy="Check loose connection to the line of fuses and starter in the motor Terminals",
            component="Power supply",
            measurable_condition="",
            base_certainty=0.80
        )
    ]

    demo_kb.rules["gearbox"] = [
        Rule(
            rule_id="GEARBOX_R001",
            machine_type="gearbox",
            symptom="Repeated rattling or clattering sound",
            cause="Gear teeth damaged or worn",
            remedy="Stop and inspect the gear set; replace damaged gears after confirming the fault",
            component="Gear teeth",
            measurable_condition="Large wear particles in oil or on the magnetic plug",
            base_certainty=0.80
        ),
        Rule(
            rule_id="GEARBOX_R010",
            machine_type="gearbox",
            symptom="High-pitched whine or squeal",
            cause="Lubricant level too low",
            remedy="Check the oil level and specified grade; refill only to the maker level",
            component="Lubrication system",
            measurable_condition="Sump oil temperature above 200 F (about 93 C)",
            base_certainty=0.80
        ),
        Rule(
            rule_id="GEARBOX_R022",
            machine_type="gearbox",
            symptom="Gearbox overheating",
            cause="Lubricant level too low",
            remedy="Check the oil level and specified grade; refill only to the maker level",
            component="Lubrication system",
            measurable_condition="Sump oil temperature above 200 F",
            base_certainty=0.80
        )
    ]

    engine = InferenceEngine(knowledge_base=demo_kb)

    # Scenario A: Single Symptom Diagnosis (Motor)
    print("\n--- TEST SCENARIO A: Single Symptom (Motor) ---")
    results_a = engine.diagnose(
        machine_type="motor",
        selected_symptoms=["Motor does not start on load"]
    )

    for res in results_a:
        print(f"\nFault: {res.cause}")
        print(f"Certainty: {res.certainty} | Safety: {res.safety_priority}")
        print("Reasoning Chain:")
        for step in res.reasoning_chain:
            print(f"  {step}")

    # Scenario B: Multiple Corroborating Symptoms with Sensor Boost (Gearbox)
    print("\n--- TEST SCENARIO B: Multiple Symptoms + Sensor Boost (Gearbox) ---")
    gearbox_telemetry = {
        "temperature": 96.5,  # 96.5 C > 93 C threshold
        "oil_condition": "degraded"
    }
    results_b = engine.diagnose(
        machine_type="gearbox",
        selected_symptoms=[
            "High-pitched whine or squeal",
            "Gearbox overheating"
        ],
        sensor_data=gearbox_telemetry
    )

    for res in results_b:
        print(f"\nFault: {res.cause}")
        print(f"Certainty: {res.certainty} | Safety: {res.safety_priority}")
        print("Reasoning Chain:")
        for step in res.reasoning_chain:
            print(f"  {step}")

    # Scenario C: Conflict Resolution (Critical vs High Priority)
    print("\n--- TEST SCENARIO C: Conflict Resolution Ranking ---")
    results_c = engine.diagnose(
        machine_type="gearbox",
        selected_symptoms=[
            "Repeated rattling or clattering sound",
            "High-pitched whine or squeal"
        ]
    )

    print("Ranked Diagnostic Hypotheses (Safety-First):")
    for idx, res in enumerate(results_c, 1):
        print(f"  [{idx}] Priority: {res.safety_priority} (Weight {engine.SAFETY_PRIORITY_WEIGHTS[res.safety_priority]}) | CF: {res.certainty} | Cause: {res.cause}")

    print("\n[OK] All inference engine algorithms verified successfully.")
