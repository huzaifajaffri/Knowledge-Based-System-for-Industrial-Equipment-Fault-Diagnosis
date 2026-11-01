"""
===============================================================================
Industrial Equipment Fault Diagnosis - Knowledge Base Module
===============================================================================

Coursework / Project: Knowledge-Based System (KBS) for Equipment Fault Diagnosis
Target Machines:
    1. Three-phase induction motor
    2. Centrifugal pump
    3. Industrial gearbox (gear reducer)
    4. Industrial air compressor

Knowledge Representation Paradigms Implemented:
    1. FRAMES (Marvin Minsky, 1974):
       Represents physical machine entities through a slot-filler structure:
       - Machine Type & Full Descriptive Name
       - Subsystem Components (structural breakdown)
       - Observable Symptoms (perceptual manifestations)
       - Diagnostic Production Rules (behavioral mapping)

    2. PRODUCTION RULES (IF-THEN Heuristic Rules):
       Antecedent (IF)   : Observed Symptom + Optional Measurable Condition
       Consequent (THEN) : Root Cause / Fault + Corrective Action / Remedy
       Metadata          : Component, Source Citation, Base Certainty Factor (CF),
                           and Safety Priority Level

Uncertainty Model:
    Shortliffe & Buchanan Certainty Factor (CF) Model (MYCIN, 1975).
    Rules carry a base CF (default = 0.8), representing expert confidence in
    the heuristic association between observed symptoms and diagnosed faults.

Safety Priority Classification:
    Derived from expert maintenance protocols (ISO 13849 / IEC 61508 concepts):
    - CRITICAL : Immediate danger, fire, explosion, shock, short-circuit, ground-fault.
    - HIGH     : Mandatory shutdown, stop operation, severe damage, replacement, burnt, trips.
    - MEDIUM   : Preventative inspection, check, calibration, monitoring, adjustment.
    - LOW      : General maintenance, minor operational variation, informational.

Author: Senior KBS Engineer & Academic Tutor
Environment: Python 3.8+ (Vanilla Python, standard library only)
===============================================================================
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Configure logging for clear debugging and execution tracing
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("KnowledgeBase")


# =============================================================================
# 1. DOMAIN CONSTANTS & ONTOLOGY MAPPINGS
# =============================================================================

# Canonical mapping of machine types to descriptive engineering titles
MACHINE_NAME_MAP: Dict[str, str] = {
    "motor": "Three-phase induction motor",
    "pump": "Centrifugal pump",
    "gearbox": "Industrial gearbox (gear reducer)",
    "compressor": "Industrial air compressor",
}

# Standard filename mappings for auto-loading CSV knowledge sources
FILENAME_MACHINE_MAP: Dict[str, str] = {
    "motor_faults.csv": "motor",
    "pump_faults.csv": "pump",
    "gearbox_faults.csv": "gearbox",
    "compressor_faults.csv": "compressor",
}

# Machine ID prefixes used for generating human-readable rule identifiers
MACHINE_ID_PREFIXES: Dict[str, str] = {
    "motor": "MOTOR",
    "pump": "PUMP",
    "gearbox": "GEARBOX",
    "compressor": "COMPRESSOR",
}

# Keyword definitions for automated Safety Priority deduction
# Evaluated strictly in order of severity: CRITICAL -> HIGH -> MEDIUM -> LOW
SAFETY_PRIORITY_KEYWORDS: Dict[str, List[str]] = {
    "critical": [
        "fire",
        "explosion",
        "electric shock",
        "immediate",
        "danger",
        "short-circuit",
        "short circuit",
        "ground-fault",
        "ground fault",
    ],
    "high": [
        "stop",
        "shut down",
        "shutdown",
        "do not",
        "severe",
        "damage",
        "replace",
        "burnt",
        "trips",
        "trip",
    ],
    "medium": [
        "inspect",
        "check",
        "monitor",
        "adjust",
    ],
}


# =============================================================================
# 2. SAFETY PRIORITY DETECTION HEURISTIC
# =============================================================================

def detect_safety_priority(cause: str, remedy: str) -> str:
    """
    Deduce safety priority ('critical', 'high', 'medium', 'low') based on
    expert domain keywords present in the cause and remedy statements.

    Why this matters in a KBS (Viva Explanation):
    ---------------------------------------------
    In an industrial setting, diagnosis is not merely an intellectual classification;
    it directs human field technicians. If a fault involves electrical shock or fire,
    the system must flag this with top priority ('critical') before less urgent
    routine inspections ('medium').

    Algorithm:
    1. Concatenate cause and remedy into a single lowercased text string.
    2. Check 'critical' keywords first. If any match as whole words/phrases, return 'critical'.
    3. Check 'high' keywords next. If any match, return 'high'.
    4. Check 'medium' keywords next. If any match, return 'medium'.
    5. Fallback to 'low' for all other cases.

    Regex word boundaries (`\\b`) are used to prevent false positives (e.g.,
    ensuring "rectifier" does not trigger "fire").

    Args:
        cause: The diagnosed fault or physical root cause.
        remedy: The corrective maintenance procedure or safety action.

    Returns:
        One of: 'critical', 'high', 'medium', 'low'.
    """
    combined_text = f"{cause} {remedy}".lower()

    # Tier 1: Critical (imminent life safety / catastrophe)
    for kw in SAFETY_PRIORITY_KEYWORDS["critical"]:
        # Word boundary pattern: allows hyphens and spaces in compound phrases
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, combined_text):
            return "critical"

    # Tier 2: High (equipment shutdown, heavy damage, trip, component replacement)
    for kw in SAFETY_PRIORITY_KEYWORDS["high"]:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, combined_text):
            return "high"

    # Tier 3: Medium (maintenance check, adjust, monitor, inspect)
    for kw in SAFETY_PRIORITY_KEYWORDS["medium"]:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, combined_text):
            return "medium"

    # Tier 4: Low (routine or informational)
    return "low"


# =============================================================================
# 3. KNOWLEDGE REPRESENTATION: RULE & FRAME DATA STRUCTURES
# =============================================================================

@dataclass
class Rule:
    """
    Represents an individual Production Rule (IF-THEN knowledge chunk).

    Theoretical Context (Viva Exam Explanation):
    --------------------------------------------
    In classical Artificial Intelligence, production rules represent heuristic
    knowledge in the form:
        IF <Antecedent / Condition> THEN <Consequent / Action>

    In this fault diagnosis KBS:
        Antecedent (LHS) : Machine Type + Symptom (+ Measurable Condition)
        Consequent (RHS) : Fault Cause + Corrective Remedy + Faulty Component

    Attributes:
        id: Unique identifier (e.g., 'MOTOR_R001', 'PUMP_R012').
        machine_type: Machine category ('motor', 'pump', 'gearbox', 'compressor').
        symptom: Qualitative perceptual symptom observed by operator (e.g. 'High vibration').
        cause: Identified root cause or engineering fault.
        remedy: Recommended corrective maintenance procedure.
        component: The physical machine component or subsystem implicated.
        measurable_condition: Quantitative sensor condition or engineering threshold (optional).
        source: Academic reference, equipment manual, or industrial standard.
        base_certainty: Prior confidence factor CF in range [0.0, 1.0] (default = 0.8).
        safety_priority: Urgency indicator ('critical', 'high', 'medium', 'low').
    """

    id: str
    machine_type: str
    symptom: str
    cause: str
    remedy: str
    component: str
    measurable_condition: str = ""
    source: str = ""
    base_certainty: float = 0.8
    safety_priority: str = "low"

    def to_dict(self) -> Dict[str, Any]:
        """Convert Rule object to a Python dictionary for JSON serialization."""
        return asdict(self)

    def matches_symptom(self, symptom_query: str) -> bool:
        """
        Check if this rule matches a query symptom.
        Performs normalized case-insensitive comparison.
        """
        clean_query = symptom_query.strip().lower()
        clean_symptom = self.symptom.strip().lower()
        return clean_query == clean_symptom or clean_query in clean_symptom


@dataclass
class Frame:
    """
    Represents a Machine Frame in Minsky's Frame Theory (1974).

    Theoretical Context (Viva Exam Explanation):
    --------------------------------------------
    A 'Frame' is an object-oriented knowledge representation structure designed
    to represent stereotypical domain entities. A frame contains:
    - SLOTS: Named properties/attributes (e.g. full_name, components, symptoms).
    - FILLERS: The values occupying those slots (e.g. specific list of parts).
    - PROCEDURAL ATTACHMENTS / RULES: Associated behavioral rules governing the entity.

    Attributes:
        machine_type: Machine identifier tag (e.g., 'motor', 'pump').
        full_name: Human-friendly descriptive name (e.g., 'Three-phase induction motor').
        components: List of unique physical components identified for this machine.
        symptoms: List of unique observable symptoms documented for this machine.
        rules: List of all Rule objects associated with this machine frame.
    """

    machine_type: str
    full_name: str
    components: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    rules: List[Rule] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert Frame to dictionary format, including nested rule dictionaries."""
        return {
            "machine_type": self.machine_type,
            "full_name": self.full_name,
            "components": list(self.components),
            "symptoms": list(self.symptoms),
            "rule_count": len(self.rules),
            "rules": [rule.to_dict() for rule in self.rules],
        }

    def get_rule_by_id(self, rule_id: str) -> Optional[Rule]:
        """Locate a specific rule inside this frame by its unique ID."""
        for rule in self.rules:
            if rule.id.upper() == rule_id.upper():
                return rule
        return None


# =============================================================================
# 4. KNOWLEDGE BASE CLASS
# =============================================================================

class KnowledgeBase:
    """
    Central Knowledge Base repository managing domain frames and production rules.

    Architecture (Viva Exam Explanation):
    -------------------------------------
    The Knowledge Base (KB) encapsulates domain-specific facts and operational rules.
    It isolates domain knowledge from the Inference Engine (the reasoning mechanism).
    This adherence to the 'Knowledge Principle' (Edward Feigenbaum) allows:
    1. Independent updating of industrial rules without altering inference logic.
    2. Modular loading of new machinery types (motor, pump, gearbox, compressor).
    3. Multi-perspective querying (by machine type, symptom, or component).
    """

    def __init__(self) -> None:
        """Initialize an empty Knowledge Base."""
        # Main storage: machine_type (lowercase string) -> Frame object
        self.frames: Dict[str, Frame] = {}
        logger.info("KnowledgeBase initialized (empty).")

    # -------------------------------------------------------------------------
    # CSV INGESTION & PARSING ENGINE
    # -------------------------------------------------------------------------

    @staticmethod
    def _clean_field(val: Optional[str]) -> str:
        """Strip surrounding whitespace and unnecessary quotation marks."""
        if val is None:
            return ""
        s = val.strip()
        # Remove residual outer quote artifacts if present
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1].strip()
        return s

    @staticmethod
    def _normalize_header(header: str) -> str:
        """Normalize CSV header strings to canonical lowercase format."""
        return re.sub(r"[\s/_-]+", " ", header.strip().lower())

    def _map_columns(self, fieldnames: List[str]) -> Dict[str, str]:
        """
        Map varied CSV header column names to standard internal field keys:
        - symptom
        - cause
        - remedy
        - component
        - measurable_condition
        - source
        """
        mapping: Dict[str, str] = {}
        for raw_col in fieldnames:
            norm = self._normalize_header(raw_col)
            if "symptom" in norm:
                mapping["symptom"] = raw_col
            elif "cause" in norm or "fault" in norm:
                mapping["cause"] = raw_col
            elif "remedy" in norm or "action" in norm:
                mapping["remedy"] = raw_col
            elif "component" in norm or "part" in norm or "subsystem" in norm:
                mapping["component"] = raw_col
            elif "measurable" in norm or "condition" in norm or "threshold" in norm:
                mapping["measurable_condition"] = raw_col
            elif "source" in norm or "reference" in norm or "citation" in norm:
                mapping["source"] = raw_col

        return mapping

    def load_machine(self, machine_type: str, csv_path: Union[str, Path]) -> Frame:
        """
        Parse an equipment CSV file, build production rules and construct a machine Frame.

        Handles:
        - Quoted multi-line fields with commas and nested quotes (RFC 4180 standard).
        - Variable UTF-8 encodings (including UTF-8 with BOM).
        - Sequential auto-generated Rule IDs (e.g. 'MOTOR_R001').
        - Automatic safety priority assignment via heuristic keyword detection.
        - De-duplicated extraction of unique components and symptoms.

        Args:
            machine_type: Category identifier ('motor', 'pump', 'gearbox', 'compressor').
            csv_path: Absolute or relative filesystem path to the CSV file.

        Returns:
            The constructed Frame object.

        Raises:
            FileNotFoundError: If the specified CSV file does not exist.
            ValueError: If mandatory columns cannot be mapped.
        """
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"Knowledge CSV not found at: {path.resolve()}")

        m_type_norm = machine_type.strip().lower()
        full_name = MACHINE_NAME_MAP.get(m_type_norm, m_type_norm.replace("_", " ").title())
        id_prefix = MACHINE_ID_PREFIXES.get(m_type_norm, m_type_norm.upper())

        logger.info(f"Loading '{m_type_norm}' rules from '{path.name}'...")

        rules: List[Rule] = []
        unique_symptoms_order: List[str] = []
        unique_components_order: List[str] = []
        seen_symptoms: Set[str] = set()
        seen_components: Set[str] = set()

        # Try utf-8-sig first (strips BOM cleanly), fallback to latin-1 if corrupted
        encodings_to_try = ["utf-8-sig", "utf-8", "latin-1"]
        raw_lines = None

        for enc in encodings_to_try:
            try:
                with open(path, mode="r", encoding=enc, newline="") as f:
                    # Test reading with csv.reader
                    reader = csv.reader(f)
                    rows = list(reader)
                    raw_lines = rows
                    break
            except UnicodeDecodeError:
                continue

        if raw_lines is None or len(raw_lines) == 0:
            raise ValueError(f"Failed to read CSV or file is completely empty: {path}")

        # Locate header row and data rows
        header_row = [self._clean_field(col) for col in raw_lines[0]]
        col_map = self._map_columns(header_row)

        # Validate minimum required columns
        for req in ["symptom", "cause", "remedy"]:
            if req not in col_map:
                raise ValueError(
                    f"CSV '{path.name}' is missing mandatory column for '{req}'. "
                    f"Found headers: {header_row}"
                )

        # Determine indices
        symptom_idx = header_row.index(col_map["symptom"])
        cause_idx = header_row.index(col_map["cause"])
        remedy_idx = header_row.index(col_map["remedy"])
        comp_idx = header_row.index(col_map["component"]) if "component" in col_map else -1
        cond_idx = (
            header_row.index(col_map["measurable_condition"])
            if "measurable_condition" in col_map
            else -1
        )
        source_idx = header_row.index(col_map["source"]) if "source" in col_map else -1

        rule_counter = 1

        for row in raw_lines[1:]:
            # Skip empty or whitespace-only rows
            if not row or all(not cell.strip() for cell in row):
                continue

            # Extract row values safely handling potential ragged rows
            def get_cell(idx: int) -> str:
                if 0 <= idx < len(row):
                    return self._clean_field(row[idx])
                return ""

            symptom_val = get_cell(symptom_idx)
            cause_val = get_cell(cause_idx)
            remedy_val = get_cell(remedy_idx)
            component_val = get_cell(comp_idx) if comp_idx != -1 else "General"
            condition_val = get_cell(cond_idx) if cond_idx != -1 else ""
            source_val = get_cell(source_idx) if source_idx != -1 else ""

            # If symptom or cause is missing, row is incomplete
            if not symptom_val or not cause_val:
                continue

            if not component_val:
                component_val = "General"

            # Auto-assign safety priority based on keyword heuristics
            safety_pri = detect_safety_priority(cause_val, remedy_val)

            # Generate standardized rule ID (e.g. MOTOR_R001)
            rule_id = f"{id_prefix}_R{rule_counter:03d}"
            rule_counter += 1

            # Instantiate Rule object with base certainty factor = 0.8
            rule_obj = Rule(
                id=rule_id,
                machine_type=m_type_norm,
                symptom=symptom_val,
                cause=cause_val,
                remedy=remedy_val,
                component=component_val,
                measurable_condition=condition_val,
                source=source_val,
                base_certainty=0.8,
                safety_priority=safety_pri,
            )
            rules.append(rule_obj)

            # Maintain de-duplicated list of symptoms preserving appearance order
            if symptom_val not in seen_symptoms:
                seen_symptoms.add(symptom_val)
                unique_symptoms_order.append(symptom_val)

            # Maintain de-duplicated list of components preserving appearance order
            if component_val and component_val not in seen_components:
                seen_components.add(component_val)
                unique_components_order.append(component_val)

        # Construct and register the Frame
        frame = Frame(
            machine_type=m_type_norm,
            full_name=full_name,
            components=unique_components_order,
            symptoms=unique_symptoms_order,
            rules=rules,
        )

        self.frames[m_type_norm] = frame
        logger.info(
            f"Successfully loaded '{m_type_norm}': "
            f"{len(rules)} rules, {len(unique_symptoms_order)} symptoms, "
            f"{len(unique_components_order)} components."
        )
        return frame

    def load_all_machines(self, dataset_dir: Union[str, Path, None] = None) -> Dict[str, Frame]:
        """
        Auto-detect and load all equipment CSV files from the specified dataset directory.

        Filename mappings:
        - 'motor_faults.csv'      -> 'motor'
        - 'pump_faults.csv'       -> 'pump'
        - 'gearbox_faults.csv'    -> 'gearbox'
        - 'compressor_faults.csv' -> 'compressor'

        If a dataset directory is not provided, defaults to the 'DataSet' folder
        co-located with this module.

        Args:
            dataset_dir: Path to directory containing the CSV fault knowledge files.

        Returns:
            Dictionary mapping machine_type -> Frame.
        """
        if dataset_dir is None:
            # Default to sibling 'DataSet' directory
            base_dir = Path(__file__).resolve().parent
            dataset_dir = base_dir / "DataSet"
        else:
            dataset_dir = Path(dataset_dir)

        if not dataset_dir.is_dir():
            logger.warning(f"DataSet directory does not exist: {dataset_dir.resolve()}")
            return self.frames

        logger.info(f"Scanning for machine fault datasets in: {dataset_dir.resolve()}")

        # Check all known filename mappings
        for filename, m_type in FILENAME_MACHINE_MAP.items():
            candidate = dataset_dir / filename
            if candidate.is_file():
                try:
                    self.load_machine(m_type, candidate)
                except Exception as ex:
                    logger.error(f"Error loading {filename} for machine '{m_type}': {ex}")
            else:
                logger.debug(f"Optional dataset '{filename}' not found at {candidate.resolve()}")

        # Also inspect directory for any files with machine names if not loaded yet
        for path in dataset_dir.glob("*.csv"):
            fname_lower = path.name.lower()
            if fname_lower in FILENAME_MACHINE_MAP:
                continue  # Already handled
            # Check if filename contains a known machine type keyword
            for m_type in MACHINE_NAME_MAP.keys():
                if m_type in fname_lower and m_type not in self.frames:
                    try:
                        self.load_machine(m_type, path)
                    except Exception as ex:
                        logger.error(f"Error loading discovered file {path.name}: {ex}")

        return self.frames

    # -------------------------------------------------------------------------
    # KNOWLEDGE BASE QUERY API
    # -------------------------------------------------------------------------

    def get_frame(self, machine_type: str) -> Optional[Frame]:
        """
        Retrieve the machine Frame object for a specified machine type.

        Args:
            machine_type: 'motor', 'pump', 'gearbox', 'compressor' (case-insensitive).

        Returns:
            Frame object if loaded, or None if machine_type is not present.
        """
        return self.frames.get(machine_type.strip().lower())

    def get_rules(self, machine_type: str) -> List[Rule]:
        """
        Retrieve all diagnostic production rules for a specified machine type.

        Args:
            machine_type: 'motor', 'pump', 'gearbox', or 'compressor'.

        Returns:
            List of Rule objects (empty list if machine not loaded).
        """
        frame = self.get_frame(machine_type)
        return list(frame.rules) if frame else []

    def get_symptoms(self, machine_type: str) -> List[str]:
        """
        Retrieve the list of unique observable symptoms for a machine type.

        Args:
            machine_type: 'motor', 'pump', 'gearbox', or 'compressor'.

        Returns:
            List of symptom description strings.
        """
        frame = self.get_frame(machine_type)
        return list(frame.symptoms) if frame else []

    def get_components(self, machine_type: str) -> List[str]:
        """
        Retrieve the list of unique components/subsystems for a machine type.

        Args:
            machine_type: 'motor', 'pump', 'gearbox', or 'compressor'.

        Returns:
            List of component name strings.
        """
        frame = self.get_frame(machine_type)
        return list(frame.components) if frame else []

    def get_machine_types(self) -> List[str]:
        """
        Retrieve a list of all currently loaded machine types.

        Returns:
            List of machine type keys, e.g. ['motor', 'pump', 'gearbox'].
        """
        return list(self.frames.keys())

    def get_rules_for_symptom(self, machine_type: str, symptom: str) -> List[Rule]:
        """
        Query all rules matching an observed symptom for a machine type.

        Matching Strategy:
        1. Exact match (case-insensitive, stripped whitespace).
        2. Fallback to substring match if no exact matches exist.

        Args:
            machine_type: Target machine identifier.
            symptom: The observed symptom string.

        Returns:
            List of matching Rule objects.
        """
        rules = self.get_rules(machine_type)
        if not rules:
            return []

        clean_query = symptom.strip().lower()

        # Step 1: Exact matches
        exact_matches = [
            r for r in rules
            if r.symptom.strip().lower() == clean_query
        ]
        if exact_matches:
            return exact_matches

        # Step 2: Substring matches (e.g. user selected partial symptom text)
        substring_matches = [
            r for r in rules
            if clean_query in r.symptom.strip().lower()
            or r.symptom.strip().lower() in clean_query
        ]
        return substring_matches

    def get_rule_by_id(self, rule_id: str) -> Optional[Rule]:
        """
        Locate any rule across all loaded machine frames using its unique ID.

        Args:
            rule_id: Unique rule ID (e.g. 'MOTOR_R005').

        Returns:
            Rule object or None if not found.
        """
        clean_id = rule_id.strip().upper()
        for frame in self.frames.values():
            for rule in frame.rules:
                if rule.id.upper() == clean_id:
                    return rule
        return None

    # -------------------------------------------------------------------------
    # SERIALIZATION & EXPORT
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the entire Knowledge Base state into a dictionary.
        Suitable for JSON HTTP API responses or client-side JavaScript delivery.

        Returns:
            Dictionary containing machine metadata, frames, rules, and statistics.
        """
        total_rules = sum(len(f.rules) for f in self.frames.values())
        total_symptoms = sum(len(f.symptoms) for f in self.frames.values())
        total_components = sum(len(f.components) for f in self.frames.values())

        return {
            "machine_types": self.get_machine_types(),
            "frames": {
                m_type: frame.to_dict()
                for m_type, frame in self.frames.items()
            },
            "stats": {
                "total_machines": len(self.frames),
                "total_rules": total_rules,
                "total_symptoms": total_symptoms,
                "total_components": total_components,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize Knowledge Base to a pretty-printed JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# =============================================================================
# 5. MODULE SELF-TEST & DEMONSTRATION RUNNER
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("KNOWLEDGE BASE MODULE - SELF TEST & VERIFICATION")
    print("=" * 80)

    kb = KnowledgeBase()
    # Resolve relative DataSet path
    current_dir = Path(__file__).resolve().parent
    dataset_path = current_dir / "DataSet"

    print(f"Loading machines from: {dataset_path}")
    kb.load_all_machines(dataset_path)

    loaded_types = kb.get_machine_types()
    print(f"\n[OK] Loaded machine types ({len(loaded_types)}): {loaded_types}")

    for m_type in loaded_types:
        frame = kb.get_frame(m_type)
        if frame:
            print(f"\n--- Machine Frame: {frame.full_name} ('{m_type}') ---")
            print(f"  * Total Rules:      {len(frame.rules)}")
            print(f"  * Unique Symptoms:  {len(frame.symptoms)}")
            print(f"  * Unique Components:{len(frame.components)}")
            if frame.rules:
                sample = frame.rules[0]
                print(f"  * Sample Rule [{sample.id}]:")
                print(f"      - Symptom:   {sample.symptom}")
                print(f"      - Cause:     {sample.cause}")
                print(f"      - Remedy:    {sample.remedy}")
                print(f"      - Component: {sample.component}")
                print(f"      - Priority:  {sample.safety_priority}")
                print(f"      - Base CF:   {sample.base_certainty}")

    # Test query API
    if "motor" in loaded_types:
        symptoms = kb.get_symptoms("motor")
        if symptoms:
            test_sym = symptoms[0]
            matched_rules = kb.get_rules_for_symptom("motor", test_sym)
            print(f"\n[OK] Query Test for Symptom: '{test_sym}'")
            print(f"     Matched {len(matched_rules)} rule(s):")
            for r in matched_rules:
                print(f"       [{r.id}] Priority: {r.safety_priority} | Cause: {r.cause}")

    # Verification of summary dictionary export
    summary = kb.to_dict()["stats"]
    print(f"\n[OK] Knowledge Base Export Summary:")
    print(f"     Machines:   {summary['total_machines']}")
    print(f"     Rules:      {summary['total_rules']}")
    print(f"     Symptoms:   {summary['total_symptoms']}")
    print(f"     Components: {summary['total_components']}")
    print("=" * 80)
