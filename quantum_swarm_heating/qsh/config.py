import os
import json
import logging
from .utils import safe_float

# Load user options
try:
    with open('/data/options.json', 'r') as f:
        user_options = json.load(f)
except Exception as e:
    logging.warning(f"Failed to load options.json: {e}. Using defaults.")
    user_options = {}

logging.info(f"Loaded user_options: {user_options}")

if not os.path.exists('/data/options.json'):
    try:
        default_config = {
            "house_rooms": {
                "lounge": 19.48,
                "open_plan": 42.14,
                "utility": 3.40,
                "cloaks": 2.51,
                "bed1": 18.17,
                "bed2": 13.59,
                "bed3": 11.07,
                "bed4": 9.79,
                "bathroom": 6.02,
                "ensuite1": 6.38,
                "ensuite2": 3.71,
                "hall": 9.15,
                "landing": 10.09
            },
            "house_facings": {
                "lounge": 0.2,
                "open_plan": 1.0,
                "utility": 0.5,
                "cloaks": 0.5,
                "bed1": 0.2,
                "bed2": 1.0,
                "bed3": 0.5,
                "bed4": 0.5,
                "bathroom": 0.2,
                "ensuite1": 0.5,
                "ensuite2": 1.0,
                "hall": 0.2,
                "landing": 0.2
            },
            "house_entities": {
                "lounge_temp_set_hum": "climate.tado_smart_radiator_thermostat_va4240580352",
                "open_plan_temp_set_hum": ["climate.tado_smart_radiator_thermostat_va0349246464", "climate.tado_smart_radiator_thermostat_va3553629184"],
                "utility_temp_set_hum": "climate.tado_smart_radiator_thermostat_va1604136448",
                "cloaks_temp_set_hum": "climate.tado_smart_radiator_thermostat_va0949825024",
                "bed1_temp_set_hum": "climate.tado_smart_radiator_thermostat_va1287620864",
                "bed2_temp_set_hum": "climate.tado_smart_radiator_thermostat_va1941512960",
                "bed3_temp_set_hum": "climate.tado_smart_radiator_thermostat_va4141228288",
                "bed4_temp_set_hum": "climate.tado_smart_radiator_thermostat_va2043158784",
                "bathroom_temp_set_hum": "climate.tado_smart_radiator_thermostat_va2920296192",
                "ensuite1_temp_set_hum": "climate.tado_smart_radiator_thermostat_va0001191680",
                "ensuite2_temp_set_hum": "climate.tado_smart_radiator_thermostat_va1209347840",
                "hall_temp_set_hum": "climate.tado_smart_radiator_thermostat_va0567183616",
                "landing_temp_set_hum": "climate.tado_smart_radiator_thermostat_va0951787776",
                "independent_sensor01": "sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor01_temperature",
                "independent_sensor02": "sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor02_temperature",
                "independent_sensor03": "sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor03_temperature",
                "independent_sensor04": "sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor04_temperature",
                "battery_soc": "sensor.givtcp_ce2029g082_soc",
                "current_day_rates": "event.octopus_energy_electricity_21l3885048_2700002762631_current_day_rates",
                "next_day_rates": "event.octopus_energy_electricity_21l3885048_2700002762631_next_day_rates",
                "current_day_export_rates": "event.octopus_energy_electricity_21l3885048_2700006856140_export_current_day_rates",
                "next_day_export_rates": "event.octopus_energy_electricity_21l3885048_2700006856140_export_next_day_rates",
                "solar_production": "sensor.envoy_122019031249_current_power_production",
                "outdoor_temp": "sensor.front_door_motion_temperature",
                "forecast_weather": "weather.home",
                "hp_output": "sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_live_heat_output",
                "hp_energy_rate": "sensor.shellyem_c4d8d5001966_channel_1_power",
                "total_heating_energy": "sensor.shellyem_c4d8d5001966_channel_1_energy",
                "water_heater": "water_heater.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31",
                "flow_min_temp": "input_number.flow_min_temperature",
                "flow_max_temp": "input_number.flow_max_temperature",
                "hp_cop": "sensor.live_cop_calc",
                "dfan_control_toggle": "input_boolean.dfan_control",
                "pid_target_temperature": "input_number.pid_target_temperature",
                "grid_power": "sensor.givtcp_ce2029g082_grid_power",
                "primary_diff": "sensor.primary_diff",
                "hp_flow_temp": "sensor.primary_flow_temperature",
                "lounge_heating": "sensor.lounge_heating",
                "open_plan_heating": "sensor.living_area_heating",
                "utility_heating": "sensor.utility_heating",
                "cloaks_heating": "sensor.wc_heating",
                "bed1_heating": "sensor.master_bedroom_heating",
                "bed2_heating": "sensor.fins_room_heating",
                "bed3_heating": "sensor.office_heating",
                "bed4_heating": "sensor.b1llz_room_heating",
                "bathroom_heating": "sensor.bathroom_heating",
                "ensuite1_heating": "sensor.ensuite1_heating",
                "ensuite2_heating": "sensor.ensuite2_heating",
                "hall_heating": "sensor.hall_heating",
                "landing_heating": "sensor.landing_heating"
            },
            "house_zone_sensor_map": {
                "hall": "independent_sensor01",
                "bed1": "independent_sensor02",
                "landing": "independent_sensor03",
                "open_plan": "independent_sensor04",
            },
            "house_battery": {
                "min_soc_reserve": 4.0,
                "efficiency": 0.9,
                "voltage": 51.0,
                "max_rate": 3.0
            },
            "house_grid": {
                "nominal_voltage": 230.0,
                "min_voltage": 200.0,
                "max_voltage": 250.0
            },
            "house_fallback_rates": {
                "cheap": 0.1495,
                "standard": 0.3048,
                "peak": 0.4572,
                "export": 0.15
            },
            "house_inverter": {
                "fallback_efficiency": 0.95
            },
            "house_peak_loss": 5.0,
            "house_design_target": 21.0,
            "house_peak_ext": -3.0,
            "house_thermal_mass_per_m2": 0.03,
            "house_heat_up_tau_h": 1.0,
            "house_persistent_zones": ["bathroom", "ensuite1", "ensuite2"],
            "house_hp_flow_service": {
                "domain": "octopus_energy",
                "service": "set_heat_pump_flow_temp_config",
                "device_id": "b680894cd18521f7c706f1305b7333ea",
                "base_data": {
                    "weather_comp_enabled": False
                }
            },
            "house_hp_hvac_service": {
                "domain": "climate",
                "service": "set_hvac_mode",
                "device_id": "b680894cd18521f7c706f1305b7333ea"
            },
            "house_room_control_mode": {
                "lounge": "direct",
                "open_plan": "indirect",
                "utility": "indirect",
                "cloaks": "indirect",
                "bed1": "indirect",
                "bed2": "indirect",
                "bed3": "indirect",
                "bed4": "indirect",
                "bathroom": "indirect",
                "ensuite1": "indirect",
                "ensuite2": "indirect",
                "hall": "indirect",
                "landing": "indirect"
            },
            "house_emitter_kw": {
                "lounge": 1.4,
                "open_plan": 3.1,
                "utility": 0.6,
                "cloaks": 0.6,
                "bed1": 1.6,
                "bed2": 1.0,
                "bed3": 1.0,
                "bed4": 1.3,
                "bathroom": 0.39,
                "ensuite1": 0.39,
                "ensuite2": 0.39,
                "hall": 1.57,
                "landing": 1.1
            },
            "house_nudge_budget": 3.0
        }
        with open('/data/options.json', 'w') as f:
            json.dump(default_config, f, indent=2)
        logging.info("Auto-created default options.json with full HOUSE_CONFIG")
    except Exception as e:
        logging.error(f"Failed to auto-create options.json: {e}")

# Define defaults
DEFAULT_ROOMS = {
    "lounge": 19.48,
    "open_plan": 42.14,
    "utility": 3.40,
    "cloaks": 2.51,
    "bed1": 18.17,
    "bed2": 13.59,
    "bed3": 11.07,
    "bed4": 9.79,
    "bathroom": 6.02,
    "ensuite1": 6.38,
    "ensuite2": 3.71,
    "hall": 9.15,
    "landing": 10.09
}
DEFAULT_FACINGS = {
    "lounge": 0.2,
    "open_plan": 1.0,
    "utility": 0.5,
    "cloaks": 0.5,
    "bed1": 0.2,
    "bed2": 1.0,
    "bed3": 0.5,
    "bed4": 0.5,
    "bathroom": 0.2,
    "ensuite1": 0.5,
    "ensuite2": 1.0,
    "hall": 0.2,
    "landing": 0.2
}
DEFAULT_ENTITIES = {
    "lounge_temp_set_hum": "climate.tado_smart_radiator_thermostat_va4240580352",
    "open_plan_temp_set_hum": ["climate.tado_smart_radiator_thermostat_va0349246464", "climate.tado_smart_radiator_thermostat_va3553629184"],
    "utility_temp_set_hum": "climate.tado_smart_radiator_thermostat_va1604136448",
    "cloaks_temp_set_hum": "climate.tado_smart_radiator_thermostat_va0949825024",
    "bed1_temp_set_hum": "climate.tado_smart_radiator_thermostat_va1287620864",
    "bed2_temp_set_hum": "climate.tado_smart_radiator_thermostat_va1941512960",
    "bed3_temp_set_hum": "climate.tado_smart_radiator_thermostat_va4141228288",
    "bed4_temp_set_hum": "climate.tado_smart_radiator_thermostat_va2043158784",
    "bathroom_temp_set_hum": "climate.tado_smart_radiator_thermostat_va2920296192",
    "ensuite1_temp_set_hum": "climate.tado_smart_radiator_thermostat_va0001191680",
    "ensuite2_temp_set_hum": "climate.tado_smart_radiator_thermostat_va1209347840",
    "hall_temp_set_hum": "climate.tado_smart_radiator_thermostat_va0567183616",
    "landing_temp_set_hum": "climate.tado_smart_radiator_thermostat_va0951787776",
    "independent_sensor01": "sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor01_temperature",
    "independent_sensor02": "sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor02_temperature",
    "independent_sensor03": "sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor03_temperature",
    "independent_sensor04": "sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor04_temperature",
    "battery_soc": "sensor.givtcp_ce2029g082_soc",
    "current_day_rates": "event.octopus_energy_electricity_21l3885048_2700002762631_current_day_rates",
    "next_day_rates": "event.octopus_energy_electricity_21l3885048_2700002762631_next_day_rates",
    "current_day_export_rates": "event.octopus_energy_electricity_21l3885048_2700006856140_export_current_day_rates",
    "next_day_export_rates": "event.octopus_energy_electricity_21l3885048_2700006856140_export_next_day_rates",
    "solar_production": "sensor.envoy_122019031249_current_power_production",
    "outdoor_temp": "sensor.front_door_motion_temperature",
    "forecast_weather": "weather.home",
    "hp_output": "sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_live_heat_output",
    "hp_energy_rate": "sensor.shellyem_c4d8d5001966_channel_1_power",
    "total_heating_energy": "sensor.shellyem_c4d8d5001966_channel_1_energy",
    "water_heater": "water_heater.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31",
    "flow_min_temp": "input_number.flow_min_temperature",
    "flow_max_temp": "input_number.flow_max_temperature",
    "hp_cop": "sensor.live_cop_calc",
    "dfan_control_toggle": "input_boolean.dfan_control",
    "pid_target_temperature": "input_number.pid_target_temperature",
    "grid_power": "sensor.givtcp_ce2029g082_grid_power",
    "primary_diff": "sensor.primary_diff",
    "hp_flow_temp": "sensor.primary_flow_temperature",
    "lounge_heating": "sensor.lounge_heating",
    "open_plan_heating": "sensor.living_area_heating",
    "utility_heating": "sensor.utility_heating",
    "cloaks_heating": "sensor.wc_heating",
    "bed1_heating": "sensor.master_bedroom_heating",
    "bed2_heating": "sensor.fins_room_heating",
    "bed3_heating": "sensor.office_heating",
    "bed4_heating": "sensor.b1llz_room_heating",
    "bathroom_heating": "sensor.bathroom_heating",
    "ensuite1_heating": "sensor.ensuite1_heating",
    "ensuite2_heating": "sensor.ensuite2_heating",
    "hall_heating": "sensor.hall_heating",
    "landing_heating": "sensor.landing_heating"
}
DEFAULT_ZONE_SENSOR_MAP = {
    "hall": "independent_sensor01",
    "bed1": "independent_sensor02",
    "landing": "independent_sensor03",
    "open_plan": "independent_sensor04",
}
DEFAULT_BATTERY = {
    "min_soc_reserve": 4.0,
    "efficiency": 0.9,
    "voltage": 51.0,
    "max_rate": 3.0
}
DEFAULT_GRID = {
    "nominal_voltage": 230.0,
    "min_voltage": 200.0,
    "max_voltage": 250.0
}
DEFAULT_FALLBACK_RATES = {
    "cheap": 0.1495,
    "standard": 0.3048,
    "peak": 0.4572,
    "export": 0.15
}
DEFAULT_INVERTER = {
    "fallback_efficiency": 0.95
}
DEFAULT_PEAK_LOSS = 5.0
DEFAULT_DESIGN_TARGET = 21.0
DEFAULT_PEAK_EXT = -3.0
DEFAULT_THERMAL_MASS_PER_M2 = 0.03
DEFAULT_HEAT_UP_TAU_H = 1.0
DEFAULT_PERSISTENT_ZONES = ["bathroom", "ensuite1", "ensuite2"]
DEFAULT_HP_FLOW_SERVICE = {
    "domain": "octopus_energy",
    "service": "set_heat_pump_flow_temp_config",
    "device_id": "b680894cd18521f7c706f1305b7333ea",
    "base_data": {
        "weather_comp_enabled": False
    }
}
DEFAULT_HP_HVAC_SERVICE = {
    "domain": "climate",
    "service": "set_hvac_mode",
    "device_id": "b680894cd18521f7c706f1305b7333ea"
}
DEFAULT_ROOM_CONTROL_MODE = {
    "lounge": "direct",
    "open_plan": "indirect",
    "utility": "indirect",
    "cloaks": "indirect",
    "bed1": "indirect",
    "bed2": "indirect",
    "bed3": "indirect",
    "bed4": "indirect",
    "bathroom": "indirect",
    "ensuite1": "indirect",
    "ensuite2": "indirect",
    "hall": "indirect",
    "landing": "indirect"
}
DEFAULT_EMITTER_KW = {
    "lounge": 1.4,
    "open_plan": 3.1,
    "utility": 0.6,
    "cloaks": 0.6,
    "bed1": 1.6,
    "bed2": 1.0,
    "bed3": 1.0,
    "bed4": 1.3,
    "bathroom": 0.39,
    "ensuite1": 0.39,
    "ensuite2": 0.39,
    "hall": 1.57,
    "landing": 1.1
}
DEFAULT_NUDGE_BUDGET = 3.0

# Parse string overrides to dict/list/float if necessary (HA may pass as strings)
def parse_override(value, default):
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if parsed:
                return parsed
            else:
                logging.warning(f"Parsed override is empty: {value}. Using default.")
                return default
        except json.JSONDecodeError:
            logging.warning(f"Failed to parse override as JSON: {value}. Using default.")
            return default
    if value is None:
        logging.warning("Override is None. Using default.")
        return default
    return value

HOUSE_CONFIG = {}
HOUSE_CONFIG['rooms'] = parse_override(user_options.get('house_rooms', DEFAULT_ROOMS), DEFAULT_ROOMS)
HOUSE_CONFIG['facings'] = parse_override(user_options.get('house_facings', DEFAULT_FACINGS), DEFAULT_FACINGS)
HOUSE_CONFIG['entities'] = parse_override(user_options.get('house_entities', DEFAULT_ENTITIES), DEFAULT_ENTITIES)
HOUSE_CONFIG['zone_sensor_map'] = parse_override(user_options.get('house_zone_sensor_map', DEFAULT_ZONE_SENSOR_MAP), DEFAULT_ZONE_SENSOR_MAP)
HOUSE_CONFIG['battery'] = parse_override(user_options.get('house_battery', DEFAULT_BATTERY), DEFAULT_BATTERY)
HOUSE_CONFIG['grid'] = parse_override(user_options.get('house_grid', DEFAULT_GRID), DEFAULT_GRID)
HOUSE_CONFIG['fallback_rates'] = parse_override(user_options.get('house_fallback_rates', DEFAULT_FALLBACK_RATES), DEFAULT_FALLBACK_RATES)
HOUSE_CONFIG['inverter'] = parse_override(user_options.get('house_inverter', DEFAULT_INVERTER), DEFAULT_INVERTER)
HOUSE_CONFIG['peak_loss'] = safe_float(user_options.get('house_peak_loss', DEFAULT_PEAK_LOSS), DEFAULT_PEAK_LOSS)
HOUSE_CONFIG['design_target'] = safe_float(user_options.get('house_design_target', DEFAULT_DESIGN_TARGET), DEFAULT_DESIGN_TARGET)
HOUSE_CONFIG['peak_ext'] = safe_float(user_options.get('house_peak_ext', DEFAULT_PEAK_EXT), DEFAULT_PEAK_EXT)
HOUSE_CONFIG['thermal_mass_per_m2'] = safe_float(user_options.get('house_thermal_mass_per_m2', DEFAULT_THERMAL_MASS_PER_M2), DEFAULT_THERMAL_MASS_PER_M2)
HOUSE_CONFIG['heat_up_tau_h'] = safe_float(user_options.get('house_heat_up_tau_h', DEFAULT_HEAT_UP_TAU_H), DEFAULT_HEAT_UP_TAU_H)
HOUSE_CONFIG['persistent_zones'] = parse_override(user_options.get('house_persistent_zones', DEFAULT_PERSISTENT_ZONES), DEFAULT_PERSISTENT_ZONES)
HOUSE_CONFIG['hp_flow_service'] = parse_override(user_options.get('house_hp_flow_service', DEFAULT_HP_FLOW_SERVICE), DEFAULT_HP_FLOW_SERVICE)
HOUSE_CONFIG['hp_hvac_service'] = parse_override(user_options.get('house_hp_hvac_service', DEFAULT_HP_HVAC_SERVICE), DEFAULT_HP_HVAC_SERVICE)
HOUSE_CONFIG['room_control_mode'] = parse_override(user_options.get('house_room_control_mode', DEFAULT_ROOM_CONTROL_MODE), DEFAULT_ROOM_CONTROL_MODE)
HOUSE_CONFIG['emitter_kw'] = parse_override(user_options.get('house_emitter_kw', DEFAULT_EMITTER_KW), DEFAULT_EMITTER_KW)
HOUSE_CONFIG['nudge_budget'] = safe_float(user_options.get('house_nudge_budget', DEFAULT_NUDGE_BUDGET), DEFAULT_NUDGE_BUDGET)

logging.info(f"Applied full HOUSE_CONFIG overrides from options.json: rooms={len(HOUSE_CONFIG['rooms'])}, entities={len(HOUSE_CONFIG['entities'])}, etc.")

# Zone offsets inferred from logs for balanced heating
ZONE_OFFSETS = {
    'lounge': 0.1,
    'open_plan': -0.1,
    'utility': 0.1,
    'cloaks': 0.1,
    'bed1': 0.6,
    'bed2': 0.6,
    'bed3': 0.4,
    'bed4': 0.4,
    'bathroom': 0.0,
    'ensuite1': 0.0,
    'ensuite2': 0.0,
    'hall': 0.1,
    'landing': 0.4
}