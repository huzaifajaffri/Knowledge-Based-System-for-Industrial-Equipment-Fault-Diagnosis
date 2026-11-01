"""
===============================================================================
KNOWLEDGE-BASED SYSTEM (KBS) FOR INDUSTRIAL EQUIPMENT FAULT DIAGNOSIS
Module: sensor_rules.py
Description: Sensor Rules & Telemetry Inference Engine based on the AI4I 2020
             Predictive Maintenance Dataset.
Author: KBS Engineering Team
Course: Knowledge Based Systems (CT-359)
===============================================================================

Theoretical Background & Viva Guide:
------------------------------------
1. Dataset Provenance:
   The threshold rules in this module are derived from statistical and domain
   analysis of the AI4I 2020 Predictive Maintenance Dataset (Dua & Graff, UCI / Matzka, 2020).
   The dataset captures 10,000 operational records of industrial equipment with
   five distinct failure modes:
   - HDF (Heat Dissipation Failure)
   - OSF (Overstrain Failure)
   - PWF (Power Failure)
   - TWF (Tool Wear Failure)
   - RNF (Random Failure)

2. Physical Rationale for Pre-Computed Thresholds:
   - HDF: Occurs when process temperature exceeds 310 K (~36.85 °C) AND the thermal
     dissipation gradient (process_temp - air_temp) is strictly greater than 8.6 K.
     This models heat exchanger fouling, fan failure, or cooling circuit blockage.
   - OSF: Occurs under excessive mechanical torque (> 60 Nm) while tool/component
     wear exceeds 200 minutes. The combination exceeds component shear and yield limits.
   - PWF: Occurs when transmitted power/torque falls outside safe operational bounds:
     Torque < 3.5 Nm (underload / coupling slippage / phase loss) OR
     Torque > 70 Nm (overload / rotor jam / impending thermal overload).
   - TWF: Occurs when tool wear enters the critical wear-out phase between 200 min
     and 240 min, leading to accelerated failure via micro-cracking and loss of tolerance.
   - RNF: An anomalous operating state where rotational speed drops sharply (< 1200 rpm)
     under heavy torque (> 55 Nm), indicating impending stall or mechanical disturbance.

3. Failure to Machine Mapping:
   Industrial machines have differing mechanical and electrical architectures:
   - HDF  -> Motor (stator winding overheating), Gearbox (oil overheating)
   - OSF  -> Motor (shaft overload), Gearbox (gear tooth overstrain), Pump (impeller jam)
   - PWF  -> Motor (electrical supply, stator winding power dissipation)
   - TWF  -> Gearbox (gear tooth abrasive wear), Pump (impeller blade erosion)
   - RNF  -> Motor, Gearbox, Pump, Compressor (stochastic disturbances)
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union


@dataclass
class SensorRule:
    """
    Data class representing a single telemetry/sensor inference rule.

    Attributes:
        rule_id: Unique identifier for the rule (e.g. 'SR01_HDF').
        failure_type: Code of failure mode ('HDF', 'OSF', 'PWF', 'TWF', 'RNF').
        description: Human-readable technical summary of the failure mode.
        condition_text: Mathematical/logical condition expression for explainability.
        certainty: Rule certainty factor CF in range [0.0, 1.0].
        relevant_machines: List of machine identifiers where this rule applies.
    """
    rule_id: str
    failure_type: str
    description: str
    condition_text: str
    certainty: float
    relevant_machines: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert the rule dataclass instance to a standard Python dictionary."""
        return asdict(self)


class SensorRuleEngine:
    """
    Rule-based inference engine that evaluates operational telemetry against
    pre-computed statistical thresholds from the AI4I 2020 dataset.
    """

    # Domain mapping connecting failure modes to industrial machine classes
    FAILURE_TO_MACHINE_MAPPING: dict[str, list[str]] = {
        'HDF': ['motor', 'gearbox'],
        'OSF': ['motor', 'gearbox', 'pump'],
        'PWF': ['motor'],
        'TWF': ['gearbox', 'pump'],
        'RNF': ['motor', 'gearbox', 'pump', 'compressor'],
    }

    def __init__(self) -> None:
        """
        Initializes the SensorRuleEngine by defining the pre-computed threshold rules
        derived from the AI4I 2020 predictive maintenance dataset analysis.
        """
        self.rules: list[SensorRule] = [
            SensorRule(
                rule_id="SR01_HDF",
                failure_type="HDF",
                description=(
                    "Heat Dissipation Failure: Elevated process temperature accompanied "
                    "by a large thermal gradient indicating heat exchanger blockage, "
                    "radiator fouling, or coolant circulation failure."
                ),
                condition_text="process_temp > 310.0 K AND (process_temp - air_temp) > 8.6 K",
                certainty=0.78,
                relevant_machines=list(self.FAILURE_TO_MACHINE_MAPPING['HDF']),
            ),
            SensorRule(
                rule_id="SR02_OSF",
                failure_type="OSF",
                description=(
                    "Overstrain Failure: Excessive mechanical torque acting upon a component "
                    "with severe cumulative wear, exceeding ultimate yield strength and "
                    "risking rapid mechanical shearing."
                ),
                condition_text="torque > 60.0 Nm AND tool_wear > 200.0 min",
                certainty=0.82,
                relevant_machines=list(self.FAILURE_TO_MACHINE_MAPPING['OSF']),
            ),
            SensorRule(
                rule_id="SR03_PWF",
                failure_type="PWF",
                description=(
                    "Power Failure: Abnormal transmitted torque indicating severe electrical "
                    "power dissipation, electrical phase failure (low torque), or motor "
                    "rotor lock / short-circuit risk (high torque)."
                ),
                condition_text="torque < 3.5 Nm OR torque > 70.0 Nm",
                certainty=0.75,
                relevant_machines=list(self.FAILURE_TO_MACHINE_MAPPING['PWF']),
            ),
            SensorRule(
                rule_id="SR04_TWF",
                failure_type="TWF",
                description=(
                    "Tool Wear Failure: Cumulative operation time has reached the critical "
                    "wear-out window (200 to 240 minutes), triggering severe clearance loss, "
                    "micro-fracturing, and surface degradation."
                ),
                condition_text="tool_wear >= 200.0 min AND tool_wear <= 240.0 min",
                certainty=0.80,
                relevant_machines=list(self.FAILURE_TO_MACHINE_MAPPING['TWF']),
            ),
            SensorRule(
                rule_id="SR05_RNF",
                failure_type="RNF",
                description=(
                    "Random Failure: Anomalous operating condition where rotational speed "
                    "drops sharply despite high mechanical resistance/torque, indicating "
                    "impending stall, foreign body obstruction, or unexpected bind."
                ),
                condition_text="rotational_speed < 1200.0 rpm AND torque > 55.0 Nm",
                certainty=0.60,
                relevant_machines=list(self.FAILURE_TO_MACHINE_MAPPING['RNF']),
            ),
        ]

        # Fast lookup cache by failure type
        self._rules_by_type: dict[str, SensorRule] = {
            rule.failure_type.upper(): rule for rule in self.rules
        }

    def _extract_float(self, data: dict, *keys: str) -> Optional[float]:
        """
        Safely extracts a float value from a dictionary given multiple possible key names.
        Handles numeric types and parseable strings, returning None on failure.
        """
        for key in keys:
            if key in data and data[key] is not None:
                val = data[key]
                if isinstance(val, (int, float)):
                    return float(val)
                val_str = str(val).strip()
                if val_str:
                    try:
                        return float(val_str)
                    except (ValueError, TypeError):
                        continue
        return None

    def evaluate(self, sensor_data: dict) -> list[SensorRule]:
        """
        Evaluates incoming sensor measurements against all pre-computed threshold rules.

        Expected sensor_data dictionary format:
        {
            'air_temp': float,          # Ambient Air temperature in Kelvin [K]
            'process_temp': float,      # Machine Process temperature in Kelvin [K]
            'rotational_speed': float,  # Operating rotational speed in [rpm]
            'torque': float,            # Shaft/drive torque in [Nm]
            'tool_wear': float          # Component/tool wear time in [min]
        }

        Returns:
            list[SensorRule]: All sensor rules whose antecedent conditions evaluated to True.
        """
        if not sensor_data or not isinstance(sensor_data, dict):
            return []

        # Extract normalized telemetry values with flexible alias fallbacks
        air_temp = self._extract_float(
            sensor_data, 'air_temp', 'air_temperature', 'air_temp_k', 'Air temperature [K]'
        )
        process_temp = self._extract_float(
            sensor_data, 'process_temp', 'process_temperature', 'process_temp_k', 'Process temperature [K]'
        )
        rotational_speed = self._extract_float(
            sensor_data, 'rotational_speed', 'speed', 'rotational_speed_rpm', 'Rotational speed [rpm]'
        )
        torque = self._extract_float(
            sensor_data, 'torque', 'torque_nm', 'Torque [Nm]'
        )
        tool_wear = self._extract_float(
            sensor_data, 'tool_wear', 'wear', 'tool_wear_min', 'Tool wear [min]'
        )

        triggered_rules: list[SensorRule] = []

        # 1. HDF Evaluation: process_temp > 310K AND (process_temp - air_temp) > 8.6K
        if process_temp is not None and air_temp is not None:
            temp_diff = process_temp - air_temp
            if process_temp > 310.0 and temp_diff > 8.6:
                rule_hdf = self._rules_by_type.get('HDF')
                if rule_hdf:
                    triggered_rules.append(rule_hdf)

        # 2. OSF Evaluation: torque > 60Nm AND tool_wear > 200min
        if torque is not None and tool_wear is not None:
            if torque > 60.0 and tool_wear > 200.0:
                rule_osf = self._rules_by_type.get('OSF')
                if rule_osf:
                    triggered_rules.append(rule_osf)

        # 3. PWF Evaluation: torque < 3.5Nm OR torque > 70Nm
        if torque is not None:
            if torque < 3.5 or torque > 70.0:
                rule_pwf = self._rules_by_type.get('PWF')
                if rule_pwf:
                    triggered_rules.append(rule_pwf)

        # 4. TWF Evaluation: tool_wear >= 200min AND tool_wear <= 240min
        if tool_wear is not None:
            if 200.0 <= tool_wear <= 240.0:
                rule_twf = self._rules_by_type.get('TWF')
                if rule_twf:
                    triggered_rules.append(rule_twf)

        # 5. RNF Evaluation: rotational_speed < 1200rpm AND torque > 55Nm
        if rotational_speed is not None and torque is not None:
            if rotational_speed < 1200.0 and torque > 55.0:
                rule_rnf = self._rules_by_type.get('RNF')
                if rule_rnf:
                    triggered_rules.append(rule_rnf)

        return triggered_rules

    def is_relevant_to_machine(self, failure_type: str, machine_type: str) -> bool:
        """
        Determines whether a given failure mode applies to a specific machine class.

        Args:
            failure_type: Failure code (e.g. 'HDF', 'OSF', 'PWF', 'TWF', 'RNF') or rule ID.
            machine_type: Target equipment identifier (e.g. 'motor', 'pump', 'gearbox', 'compressor').

        Returns:
            bool: True if the failure mode is physically relevant to the machine; False otherwise.
        """
        if not failure_type or not machine_type:
            return False

        # Clean and normalize strings
        ft_clean = failure_type.strip().upper()
        # If full rule_id was passed (e.g. 'SR01_HDF'), extract failure_type suffix
        if '_' in ft_clean:
            ft_clean = ft_clean.split('_')[-1]

        mt_clean = machine_type.strip().lower()

        relevant_machines = self.FAILURE_TO_MACHINE_MAPPING.get(ft_clean, [])
        return mt_clean in [m.lower() for m in relevant_machines]

    def get_all_rules_description(self) -> list[dict[str, Any]]:
        """
        Returns serializable metadata descriptions of all defined sensor rules.

        Returns:
            list[dict]: List of rule metadata dictionaries suitable for JSON API responses.
        """
        return [rule.to_dict() for rule in self.rules]

    def evaluate_for_machine(self, sensor_data: dict, machine_type: str) -> list[SensorRule]:
        """
        Evaluates sensor rules and filters results strictly to those physically
        relevant to the specified machine type.

        Args:
            sensor_data: Dictionary of telemetry measurements.
            machine_type: Target machine identifier ('motor', 'gearbox', 'pump', etc.).

        Returns:
            list[SensorRule]: Triggered rules that are relevant to the machine.
        """
        triggered = self.evaluate(sensor_data)
        return [r for r in triggered if self.is_relevant_to_machine(r.failure_type, machine_type)]

    def get_rules_for_machine(self, machine_type: str) -> list[SensorRule]:
        """
        Retrieves all rules defined as applicable to a given machine type.
        """
        mt_clean = machine_type.strip().lower()
        return [
            rule for rule in self.rules
            if any(m.lower() == mt_clean for m in rule.relevant_machines)
        ]
