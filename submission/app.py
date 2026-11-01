from __future__ import annotations
import logging
import os
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request

# Import KBS Core Modules
from knowledge_base import KnowledgeBase
from inference_engine import InferenceEngine
from sensor_rules import SensorRuleEngine

# Configure structured logging for API tracing and viva demonstration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("KBS_App")

# =============================================================================
# FLASK APPLICATION CONFIGURATION
# =============================================================================

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

# Application runtime configuration
app.config["DEBUG"] = True
app.config["JSON_SORT_KEYS"] = False

# Resolve absolute path to the DataSet directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "DataSet")

# =============================================================================
# KNOWLEDGE BASE & INFERENCE ENGINES INITIALIZATION
# =============================================================================

logger.info(f"Initializing Knowledge Base from DataSet directory: {DATASET_DIR}")
kb = KnowledgeBase()

# Load all machine fault knowledge files from DataSet/
try:
    kb.load_all_machines(DATASET_DIR)
    logger.info(f"Loaded machines into KnowledgeBase: {kb.get_machine_types()}")
except Exception as ex:
    logger.error(f"Error loading machines into KnowledgeBase: {ex}", exc_info=True)

# Initialize the forward chaining inference engine with the loaded Knowledge Base
inference_engine = InferenceEngine(kb)
logger.info("InferenceEngine initialized successfully.")

# Initialize the AI4I 2020 sensor rule engine
sensor_rule_engine = SensorRuleEngine()
logger.info("SensorRuleEngine initialized successfully.")

# =============================================================================
# HELPER DOMAIN DEFINITIONS FOR SENSOR CORROBORATION & BOOSTING
# =============================================================================

# Keyword associations between sensor failure types and mechanical/electrical concepts
# Used to determine semantic relevance between sensor alerts and symptom diagnoses
SENSOR_FAILURE_KEYWORDS: Dict[str, List[str]] = {
    "HDF": [
        "heat", "temp", "thermal", "cooling", "coolant", "fan", "overheat",
        "hot", "ventilat", "radiator", "lubricant film", "viscosity", "oil temperature", "burn"
    ],
    "OSF": [
        "overload", "torque", "jam", "strain", "bind", "seize", "stuck",
        "high load", "inertia", "stress", "heavy", "tight", "friction", "mechanical"
    ],
    "PWF": [
        "power", "voltage", "electrical", "fuse", "breaker", "phase", "starter",
        "relay", "current", "wire", "cable", "terminal", "unbalance", "trip", "supply"
    ],
    "TWF": [
        "wear", "worn", "erosion", "clearance", "backlash", "play", "fatigue",
        "damage", "chipped", "spalling", "pitting", "scoring", "teeth", "impeller blade"
    ],
    "RNF": [
        "speed", "stall", "slip", "slow", "erratic", "intermittent", "foreign",
        "debris", "vibrat", "noise", "knocking", "rattle", "obstruction"
    ],
}

# Default maintenance remedies prescribed for direct telemetry anomaly alerts
SENSOR_DEFAULT_REMEDIES: Dict[str, str] = {
    "HDF": (
        "Inspect cooling fan, airflow ducts, and heat exchanger fins for blockage. "
        "Verify coolant/oil circulation rate and examine thermal dissipation integrity."
    ),
    "OSF": (
        "Immediately relieve mechanical load. Inspect drive train for mechanical binding, "
        "jammed shafts, or severe component wear exceeding safe structural limits."
    ),
    "PWF": (
        "Verify three-phase electrical supply voltage and balance. Inspect circuit breakers, "
        "contactors, overload relays, and motor terminal lugs for loose connections."
    ),
    "TWF": (
        "Schedule prompt component replacement. Inspect gear teeth contact patterns, "
        "bearing clearances, or pump impeller wear rings for catastrophic wear-out."
    ),
    "RNF": (
        "Perform immediate mechanical inspection for foreign object debris or sudden shaft "
        "stall. Verify rotational speed sensor accuracy and inspect flexible coupling."
    ),
}

# Default component associations for direct telemetry alerts
SENSOR_DEFAULT_COMPONENTS: Dict[str, Dict[str, str]] = {
    "HDF": {
        "motor": "Cooling fan & stator core",
        "gearbox": "Gear oil & housing",
        "pump": "Pump bearing housing & cooling jacket",
        "compressor": "Intercooler & cylinder head",
    },
    "OSF": {
        "motor": "Motor shaft & driven equipment",
        "gearbox": "Gear mesh & pinion shaft",
        "pump": "Pump impeller & shaft",
        "compressor": "Compressor crankshaft & piston assembly",
    },
    "PWF": {
        "motor": "Power supply & terminal box",
        "gearbox": "Electric drive coupling",
        "pump": "Motor phase connections",
        "compressor": "Starter switchgear & motor supply",
    },
    "TWF": {
        "motor": "Drive-end bearings",
        "gearbox": "Gear teeth & journal bearings",
        "pump": "Impeller blades & wear rings",
        "compressor": "Piston rings & cylinder sleeve",
    },
    "RNF": {
        "motor": "Rotor assembly & bearing cage",
        "gearbox": "Gear transmission train",
        "pump": "Impeller & shaft sleeve",
        "compressor": "Suction/discharge valves & crosshead",
    },
}

# =============================================================================
# CORS MIDDLEWARE & REQUEST HOOKS
# =============================================================================

@app.after_request
def add_cors_headers(response):
    """
    Appends standard Cross-Origin Resource Sharing (CORS) headers to all HTTP responses.
    Allows decoupling frontend development (live-servers) from the Flask backend.
    """
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


@app.route("/api/<path:subpath>", methods=["OPTIONS"])
def api_options_handler(subpath: str):
    """Handles HTTP preflight OPTIONS requests from web browsers."""
    return ("", 204)


# =============================================================================
# REST API ENDPOINTS
# =============================================================================

@app.route("/", methods=["GET"])
def index():
    """
    Endpoint 1: Root route serving the main user interface.
    Serves `templates/index.html` or provides an informative HTML status page
    if the frontend template has not yet been rendered.
    """
    try:
        return render_template("index.html")
    except Exception as ex:
        logger.warning(f"Could not render templates/index.html: {ex}")
        return (
            "<!DOCTYPE html>"
            "<html lang='en'>"
            "<head>"
            "  <meta charset='UTF-8'>"
            "  <title>KBS Fault Diagnosis System</title>"
            "  <style>"
            "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; "
            "           background: #0a0e1a; color: #f1f5f9; padding: 3rem; line-height: 1.6; }"
            "    h1 { color: #60a5fa; margin-bottom: 0.5rem; }"
            "    .card { background: rgba(17, 24, 39, 0.8); border: 1px solid rgba(255,255,255,0.1); "
            "            border-radius: 12px; padding: 1.5rem; max-width: 700px; margin-top: 1.5rem; }"
            "    code { background: rgba(0,0,0,0.4); padding: 0.2rem 0.4rem; border-radius: 4px; color: #06b6d4; }"
            "    a { color: #60a5fa; text-decoration: none; }"
            "    a:hover { text-decoration: underline; }"
            "    ul { margin-top: 0.5rem; padding-left: 1.5rem; }"
            "  </style>"
            "</head>"
            "<body>"
            "  <h1>Industrial Equipment Fault Diagnosis KBS</h1>"
            "  <p>Flask Backend API Server is active and operational.</p>"
            "  <div class='card'>"
            "    <h3>Active REST API Endpoints:</h3>"
            "    <ul>"
            "      <li><a href='/api/machines'><code>GET /api/machines</code></a> - Loaded machinery & stats</li>"
            "      <li><code>GET /api/machines/&lt;machine_type&gt;/symptoms</code> - Observable symptoms</li>"
            "      <li><code>GET /api/machines/&lt;machine_type&gt;/components</code> - Subsystem components</li>"
            "      <li><code>POST /api/diagnose</code> - Forward chaining diagnosis endpoint</li>"
            "      <li><a href='/api/sensor-rules'><code>GET /api/sensor-rules</code></a> - AI4I 2020 telemetry rules</li>"
            "    </ul>"
            "  </div>"
            "</body>"
            "</html>"
        ), 200


@app.route("/api/machines", methods=["GET"])
def get_machines():
    """
    Endpoint 2: Retrieve all loaded machines along with their quantitative statistics.

    Returns:
        JSON list of machine objects with:
        - id / machine_type
        - full descriptive title
        - symptom_count
        - rule_count
        - component_count
    """
    try:
        machines_data: list[dict[str, Any]] = []

        # Canonical machine metadata dictionary for consistent ordering and naming
        canonical_machines = ["motor", "pump", "gearbox", "compressor"]

        # First collect all frames loaded in KnowledgeBase
        for m_type in canonical_machines:
            frame = kb.get_frame(m_type)
            if frame:
                machines_data.append({
                    "id": m_type,
                    "machine_type": m_type,
                    "name": frame.full_name,
                    "symptom_count": len(frame.symptoms),
                    "rule_count": len(frame.rules),
                    "component_count": len(frame.components),
                })

        # Append any additional machine frames loaded beyond canonical four
        for m_type, frame in kb.frames.items():
            if m_type not in canonical_machines:
                machines_data.append({
                    "id": m_type,
                    "machine_type": m_type,
                    "name": frame.full_name,
                    "symptom_count": len(frame.symptoms),
                    "rule_count": len(frame.rules),
                    "component_count": len(frame.components),
                })

        return jsonify(machines_data), 200

    except Exception as ex:
        logger.error(f"Error in GET /api/machines: {ex}", exc_info=True)
        return jsonify({"error": "Failed to retrieve machine list", "details": str(ex)}), 500


@app.route("/api/machines/<machine_type>/symptoms", methods=["GET"])
def get_machine_symptoms(machine_type: str):
    """
    Endpoint 3: Returns unique observable symptoms for a specified machine.

    Args:
        machine_type: 'motor', 'pump', 'gearbox', or 'compressor' (case-insensitive).

    Returns:
        JSON array of unique symptom strings.
    """
    try:
        m_norm = machine_type.strip().lower()
        frame = kb.get_frame(m_norm)

        if not frame:
            return jsonify({
                "error": f"Machine '{machine_type}' was not found in the Knowledge Base.",
                "available_machines": kb.get_machine_types()
            }), 404

        symptoms = list(frame.symptoms)
        return jsonify(symptoms), 200

    except Exception as ex:
        logger.error(f"Error in GET /api/machines/{machine_type}/symptoms: {ex}", exc_info=True)
        return jsonify({"error": "Failed to retrieve symptoms", "details": str(ex)}), 500


@app.route("/api/machines/<machine_type>/components", methods=["GET"])
def get_machine_components(machine_type: str):
    """
    Endpoint 4: Returns unique subsystem components for a specified machine.

    Args:
        machine_type: 'motor', 'pump', 'gearbox', or 'compressor' (case-insensitive).

    Returns:
        JSON array of unique component strings.
    """
    try:
        m_norm = machine_type.strip().lower()
        frame = kb.get_frame(m_norm)

        if not frame:
            return jsonify({
                "error": f"Machine '{machine_type}' was not found in the Knowledge Base.",
                "available_machines": kb.get_machine_types()
            }), 404

        components = list(frame.components)
        return jsonify(components), 200

    except Exception as ex:
        logger.error(f"Error in GET /api/machines/{machine_type}/components: {ex}", exc_info=True)
        return jsonify({"error": "Failed to retrieve components", "details": str(ex)}), 500


@app.route("/api/sensor-rules", methods=["GET"])
def get_sensor_rules():
    """
    Endpoint 6: Returns metadata descriptions of all AI4I 2020 telemetry threshold rules.

    Returns:
        JSON list of sensor rule dictionaries.
    """
    try:
        rules_desc = sensor_rule_engine.get_all_rules_description()
        return jsonify(rules_desc), 200
    except Exception as ex:
        logger.error(f"Error in GET /api/sensor-rules: {ex}", exc_info=True)
        return jsonify({"error": "Failed to retrieve sensor rules", "details": str(ex)}), 500


@app.route("/api/diagnose", methods=["POST"])
def diagnose():
    """
    Endpoint 5: Main diagnostic inference endpoint.

    Workflow:
    1. Validate request body and extract machine_type, symptoms, and optional sensor_data.
    2. Execute Forward Chaining reasoning in InferenceEngine with observed symptoms.
    3. If sensor telemetry data is provided:
       a. Evaluate telemetry against pre-computed AI4I 2020 threshold rules.
       b. Filter triggered rules strictly for relevance to the target machine.
       c. Boost certainty of related diagnoses using MYCIN combination algebra.
       d. Synthesize direct telemetry diagnoses if telemetry indicates an independent fault.
    4. Return the combined, ranked list of diagnosis results with explanations.

    Request Body Format:
    {
        "machine_type": "motor",
        "symptoms": ["Motor does not start on load."],
        "sensor_data": {
            "air_temp": 300.0,
            "process_temp": 312.0,
            "rotational_speed": 1150.0,
            "torque": 65.0,
            "tool_wear": 210.0
        }
    }

    Returns:
        JSON array of sorted diagnosis results with certainty, remedies, and explanations.
    """
    # Step 1: Input parsing and robust validation
    payload = request.get_json(silent=True)
    if not payload or not isinstance(payload, dict):
        return jsonify({
            "error": "Invalid request. Expected a JSON object with 'machine_type' and 'symptoms'."
        }), 400

    machine_type_raw = payload.get("machine_type")
    if not machine_type_raw or not str(machine_type_raw).strip():
        return jsonify({"error": "Field 'machine_type' is mandatory."}), 400

    machine_type = str(machine_type_raw).strip().lower()

    # Verify that the machine exists in the Knowledge Base
    if not kb.get_frame(machine_type):
        return jsonify({
            "error": f"Machine '{machine_type}' is not recognized or loaded in the Knowledge Base.",
            "available_machines": kb.get_machine_types()
        }), 404

    # Extract symptoms list safely
    raw_symptoms = payload.get("symptoms", [])
    if isinstance(raw_symptoms, str):
        symptoms = [raw_symptoms.strip()] if raw_symptoms.strip() else []
    elif isinstance(raw_symptoms, list):
        symptoms = [str(s).strip() for s in raw_symptoms if s and str(s).strip()]
    else:
        symptoms = []

    # Extract optional sensor telemetry dictionary
    sensor_data = payload.get("sensor_data")
    if sensor_data is not None and not isinstance(sensor_data, dict):
        sensor_data = None

    # Enforce that the user provided at least one symptom or sensor measurement
    if not symptoms and not sensor_data:
        return jsonify({
            "error": "At least one observed symptom or sensor measurement must be provided for diagnosis."
        }), 400

    logger.info(
        f"Diagnose request received: machine='{machine_type}', "
        f"symptoms={len(symptoms)}, sensor_data_present={sensor_data is not None}"
    )

    try:
        # Step 2: Run forward chaining inference engine on observed symptoms
        # Note: InferenceEngine returns a list of DiagnosisResult dataclass instances
        raw_inferred_results = inference_engine.diagnose(
            machine_type=machine_type,
            selected_symptoms=symptoms,
            sensor_data=sensor_data
        )

        # Standardize results into structured dictionaries
        combined_diagnoses: list[dict[str, Any]] = []
        for item in raw_inferred_results:
            cause_text = getattr(item, "cause", "")
            remedy_text = getattr(item, "remedy", "")
            comp_text = getattr(item, "component", "")
            certainty_val = float(getattr(item, "certainty", 0.80))
            priority_val = str(getattr(item, "safety_priority", "MEDIUM")).upper()
            matched_syms = list(getattr(item, "matched_symptoms", []))
            matched_rules = list(getattr(item, "matched_rules", []))
            reasoning = list(getattr(item, "reasoning_chain", []))

            # Provide comprehensive diagnostic representation
            diag_dict = {
                "cause": cause_text,
                "fault": cause_text,  # Alias for client compatibility
                "remedy": remedy_text,
                "component": comp_text,
                "certainty": round(certainty_val, 4),
                "safety_priority": priority_val,
                "matched_symptoms": matched_syms,
                "matched_rules": matched_rules,
                "reasoning_chain": reasoning,
                "explanation": (
                    reasoning[0] if reasoning else f"Diagnosed from symptom: {', '.join(matched_syms)}"
                ),
                "sensor_supported": False,
                "sensor_evidence": [],
                "source": "Empirical Equipment Knowledge Base",
            }
            combined_diagnoses.append(diag_dict)

        # Step 3: Evaluate sensor telemetry rules if sensor_data was provided
        relevant_sensor_rules = []
        if sensor_data and isinstance(sensor_data, dict):
            triggered_rules = sensor_rule_engine.evaluate(sensor_data)
            # Filter strictly for rules relevant to the current machine type
            relevant_sensor_rules = [
                r for r in triggered_rules
                if sensor_rule_engine.is_relevant_to_machine(r.failure_type, machine_type)
            ]
            logger.info(
                f"Sensor rules evaluated: {len(triggered_rules)} triggered, "
                f"{len(relevant_sensor_rules)} relevant to machine '{machine_type}'"
            )

        # Step 4: Sensor Corroboration & Certainty Boosting
        # For each triggered and relevant sensor rule, boost related symptom diagnoses
        # or assert direct telemetry alerts if the condition represents an independent failure
        applied_sensor_failures = set()

        if relevant_sensor_rules:
            for s_rule in relevant_sensor_rules:
                f_type = s_rule.failure_type.upper()
                keywords = SENSOR_FAILURE_KEYWORDS.get(f_type, [])
                rule_matched_any_diag = False

                for diag in combined_diagnoses:
                    # Check if diagnosis text relates to the sensor failure mode
                    search_corpus = (
                        f"{diag['cause']} {diag['component']} {' '.join(diag['matched_symptoms'])}"
                    ).lower()

                    is_related = any(kw in search_corpus for kw in keywords)

                    if is_related:
                        rule_matched_any_diag = True
                        applied_sensor_failures.add(f_type)

                        # Apply Shortliffe & Buchanan MYCIN Certainty Combination:
                        # CF_boosted = CF_old + (1 - CF_old) * (CF_sensor * 0.5)
                        old_cf = diag["certainty"]
                        effective_sensor_weight = s_rule.certainty * 0.5
                        combined_cf = old_cf + (1.0 - old_cf) * effective_sensor_weight
                        new_cf = min(0.99, round(combined_cf, 4))

                        diag["certainty"] = new_cf
                        diag["sensor_supported"] = True
                        diag["sensor_evidence"].append(s_rule.to_dict())

                        # Append explicit transparent justification to the explanation trace
                        boost_step = (
                            f"Sensor Telemetry Corroboration: Certainty boosted from "
                            f"{int(old_cf * 100)}% to {int(new_cf * 100)}% by Rule {s_rule.rule_id} "
                            f"({s_rule.failure_type}) - Condition: [{s_rule.condition_text}]."
                        )
                        diag["reasoning_chain"].append(boost_step)
                        diag["explanation"] += f" Corroborated by telemetry rule {s_rule.rule_id}."

                # If the sensor rule triggered but was NOT tied to an existing symptom diagnosis,
                # create a distinct direct telemetry diagnosis
                if not rule_matched_any_diag:
                    comp_name = SENSOR_DEFAULT_COMPONENTS.get(f_type, {}).get(
                        machine_type, f"{machine_type.title()} Critical Subsystem"
                    )
                    remedy_text = SENSOR_DEFAULT_REMEDIES.get(
                        f_type, "Inspect equipment immediately and perform preventative maintenance."
                    )
                    fault_title = f"Telemetry Alert: {s_rule.description.split(':')[0]} ({f_type})"

                    telemetry_diag = {
                        "cause": fault_title,
                        "fault": fault_title,
                        "remedy": remedy_text,
                        "component": comp_name,
                        "certainty": round(s_rule.certainty, 4),
                        "safety_priority": "CRITICAL" if s_rule.certainty >= 0.80 else "HIGH",
                        "matched_symptoms": [],
                        "matched_rules": [s_rule.rule_id],
                        "reasoning_chain": [
                            f"Telemetry Observation: Operational sensor inputs satisfied condition: {s_rule.condition_text}.",
                            f"Domain Validation: Failure mode '{f_type}' verified applicable to {machine_type}.",
                            f"Direct Telemetry Deduction: Root cause asserted with sensor rule certainty {s_rule.certainty}.",
                        ],
                        "explanation": (
                            f"Direct telemetry deduction via Rule {s_rule.rule_id}: "
                            f"Condition [{s_rule.condition_text}] met."
                        ),
                        "sensor_supported": True,
                        "sensor_evidence": [s_rule.to_dict()],
                        "source": "AI4I 2020 Predictive Maintenance Dataset Telemetry",
                    }
                    combined_diagnoses.append(telemetry_diag)

        # Step 5: Rank results by certainty factor in descending order
        # Secondary sort key maintains safety priority ordering (CRITICAL > HIGH > MEDIUM > LOW)
        priority_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}

        combined_diagnoses.sort(
            key=lambda d: (
                d.get("certainty", 0.0),
                priority_order.get(d.get("safety_priority", "LOW"), 1)
            ),
            reverse=True
        )

        return jsonify(combined_diagnoses), 200

    except Exception as ex:
        logger.error(f"Error during diagnostic inference: {ex}", exc_info=True)
        return jsonify({
            "error": "Diagnostic inference failed due to an internal calculation error.",
            "details": str(ex)
        }), 500


# =============================================================================
# GENERAL ERROR HANDLERS
# =============================================================================

@app.errorhandler(400)
def bad_request_handler(error):
    """Handles 400 Bad Request HTTP errors."""
    return jsonify({
        "error": "Bad Request",
        "message": str(getattr(error, "description", error))
    }), 400


@app.errorhandler(404)
def not_found_handler(error):
    """Handles 404 Not Found HTTP errors."""
    return jsonify({
        "error": "Resource Not Found",
        "message": str(getattr(error, "description", error))
    }), 404


@app.errorhandler(500)
def internal_server_error_handler(error):
    """Handles 500 Internal Server Errors."""
    logger.error(f"HTTP 500 handler triggered: {error}")
    return jsonify({
        "error": "Internal Server Error",
        "message": "An unhandled exception occurred in the KBS backend server."
    }), 500


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    logger.info("Starting KBS Fault Diagnosis Web Server on http://0.0.0.0:5000 ...")
    app.run(debug=True, host='0.0.0.0', port=5000)
