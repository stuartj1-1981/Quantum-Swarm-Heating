from logging import config
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from datetime import datetime, timedelta, timezone
import time
import logging
import os
import json
import requests
from collections import defaultdict, deque
import signal
import sys
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load user options
try:
    with open('/data/options.json', 'r') as f:
        user_options = json.load(f)
except Exception as e:
    logging.warning(f"Failed to load options.json: {e}. Using defaults.")
    user_options = {}

logging.info(f"Loaded user_options: {user_options}")

# HA API setup
HA_URL = 'http://supervisor/core/api'
HA_TOKEN = os.getenv('SUPERVISOR_TOKEN')

logging.info(f"Detected SUPERVISOR_TOKEN: {'Set' if HA_TOKEN else 'None'}")

if not HA_TOKEN:
    logging.critical("SUPERVISOR_TOKEN not set! Using defaults only. Ensure hassio_api: true in config.yaml.")
else:
    logging.info("SUPERVISOR_TOKEN found—using real HA API calls.")

def fetch_ha_entity(entity_id, attr=None):
    if not HA_TOKEN:
        return None
    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json"
    }
    try:
        r = requests.get(f"{HA_URL}/states/{entity_id}", headers=headers)
        r.raise_for_status()
        data = r.json()
        if attr:
            return data.get('attributes', {}).get(attr)
        return data.get('state')
    except Exception as e:
        logging.error(f"HA fetch error for {entity_id}: {e}")
        return None

def set_ha_service(domain, service, data):
    if not HA_TOKEN:
        return
    headers = {'Authorization': f"Bearer {HA_TOKEN}"}
    entity_id = data.get('entity_id')
    retries = 0
    max_retries = 3
    while retries < max_retries:
        try:
            if isinstance(entity_id, list):
                for eid in entity_id:
                    data_single = data.copy()
                    data_single['entity_id'] = eid
                    r = requests.post(f"{HA_URL}/services/{domain}/{service}", headers=headers, json=data_single)
                    r.raise_for_status()
            else:
                r = requests.post(f"{HA_URL}/services/{domain}/{service}", headers=headers, json=data)
                r.raise_for_status()
            return  # Success
        except Exception as e:
            retries += 1
            logging.warning(f"HA set error for {entity_id or data.get('device_id')}: {e} - retry {retries}/{max_retries}")
            time.sleep(5)  # 5s sleep
    logging.error(f"HA set failed after {max_retries} retries for {entity_id or data.get('device_id')}")

# Default HOUSE_CONFIG
DEFAULT_ROOMS = { 'lounge': 19.48, 'open_plan': 42.14, 'utility': 3.40, 'cloaks': 2.51,
    'bed1': 18.17, 'bed2': 13.59, 'bed3': 11.07, 'bed4': 9.79, 'bathroom': 6.02, 'ensuite1': 6.38, 'ensuite2': 3.71,
    'hall': 9.15, 'landing': 10.09 }
DEFAULT_FACINGS = { 'lounge': 0.2, 'open_plan': 1.0, 'utility': 0.5, 'cloaks': 0.5,
    'bed1': 0.2, 'bed2': 1.0, 'bed3': 0.5, 'bed4': 0.5, 'bathroom': 0.2, 'ensuite1': 0.5, 'ensuite2': 1.0,
    'hall': 0.2, 'landing': 0.2 }
DEFAULT_ENTITIES = {
    'lounge_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va4240580352',
    'open_plan_temp_set_hum': ['climate.tado_smart_radiator_thermostat_va0349246464', 'climate.tado_smart_radiator_thermostat_va3553629184'],
    'utility_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va1604136448',
    'cloaks_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va0949825024',
    'bed1_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va1287620864',
    'bed2_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va1941512960',
    'bed3_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va4141228288',
    'bed4_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va2043158784',
    'bathroom_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va2920296192',
    'ensuite1_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va0001191680',
    'ensuite2_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va1209347840',
    'hall_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va0567183616',
    'landing_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va0951787776',
    'independent_sensor01': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor01_temperature',
    'independent_sensor02': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor02_temperature',
    'independent_sensor03': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor03_temperature',
    'independent_sensor04': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor04_temperature',
    'battery_soc': 'sensor.givtcp_ce2029g082_soc',
    'current_day_rates': 'event.octopus_energy_electricity_21l3885048_2700002762631_current_day_rates',
    'next_day_rates': 'event.octopus_energy_electricity_21l3885048_2700002762631_next_day_rates',
    'current_day_export_rates': 'event.octopus_energy_electricity_21l3885048_2700006856140_export_current_day_rates',
    'next_day_export_rates': 'event.octopus_energy_electricity_21l3885048_2700006856140_export_next_day_rates',
    'solar_production': 'sensor.envoy_122019031249_current_power_production',
    'outdoor_temp': 'sensor.front_door_motion_temperature',
    'forecast_weather': 'weather.home',
    'hp_output': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_live_heat_output',
    'hp_energy_rate': 'sensor.shellyem_c4d8d5001966_channel_1_power',
    'total_heating_energy': 'sensor.shellyem_c4d8d5001966_channel_1_energy',
    'water_heater': 'water_heater.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31',
    'flow_min_temp': 'input_number.flow_min_temperature',
    'flow_max_temp': 'input_number.flow_max_temperature',
    'hp_cop': 'sensor.live_cop_calc',
    'dfan_control_toggle': 'input_boolean.dfan_control',
    'pid_target_temperature': 'input_number.pid_target_temperature',
    'grid_power': 'sensor.givtcp_ce2029g082_grid_power',
    'primary_diff': 'sensor.primary_diff',
    'hp_flow_temp': 'sensor.primary_flow_temperature',
    'lounge_heating': 'sensor.lounge_heating',
    'open_plan_heating': 'sensor.living_area_heating',
    'utility_heating': 'sensor.utility_heating',
    'cloaks_heating': 'sensor.wc_heating',
    'bed1_heating': 'sensor.master_bedroom_heating',
    'bed2_heating': 'sensor.fins_room_heating',
    'bed3_heating': 'sensor.office_heating',
    'bed4_heating': 'sensor.b1llz_room_heating',
    'bathroom_heating': 'sensor.bathroom_heating',
    'ensuite1_heating': 'sensor.ensuite1_heating',
    'ensuite2_heating': 'sensor.ensuite2_heating',
    'hall_heating': 'sensor.hall_heating',
    'landing_heating': 'sensor.landing_heating'
}
DEFAULT_ZONE_SENSOR_MAP = { 'hall': 'independent_sensor01', 'bed1': 'independent_sensor02', 'landing': 'independent_sensor03', 'open_plan': 'independent_sensor04',
    'utility': 'independent_sensor01', 'cloaks': 'independent_sensor01', 'bed2': 'independent_sensor02', 'bed3': 'independent_sensor03', 'bed4': 'independent_sensor03',
    'bathroom': 'independent_sensor03', 'ensuite1': 'independent_sensor02', 'ensuite2': 'independent_sensor03', 'lounge': 'independent_sensor01' }
DEFAULT_BATTERY = {'min_soc_reserve': 4.0, 'efficiency': 0.9, 'voltage': 51.0, 'max_rate': 3.0}
DEFAULT_GRID = {'nominal_voltage': 230.0, 'min_voltage': 200.0, 'max_voltage': 250.0}
DEFAULT_FALLBACK_RATES = {'cheap': 0.1495, 'standard': 0.3048, 'peak': 0.4572, 'export': 0.15}
DEFAULT_INVERTER = {'fallback_efficiency': 0.95}
DEFAULT_PEAK_LOSS = 5.0
DEFAULT_DESIGN_TARGET = 21.0
DEFAULT_PEAK_EXT = -3.0
DEFAULT_THERMAL_MASS_PER_M2 = 0.03
DEFAULT_HEAT_UP_TAU_H = 1.0
DEFAULT_PERSISTENT_ZONES = ['bathroom', 'ensuite1', 'ensuite2']
DEFAULT_HP_FLOW_SERVICE = {
    'domain': 'octopus_energy',
    'service': 'set_heat_pump_flow_temp_config',
    'device_id': 'b680894cd18521f7c706f1305b7333ea',
    'base_data': {
        'weather_comp_enabled': False
    }
}
DEFAULT_HP_HVAC_SERVICE = {
    'domain': 'climate',
    'service': 'set_hvac_mode',
    'device_id': 'b680894cd18521f7c706f1305b7333ea'
}
DEFAULT_ROOM_CONTROL_MODE = {
    'lounge': 'direct',
    'open_plan': 'indirect',
    'utility': 'indirect',
    'cloaks': 'indirect',
    'bed1': 'indirect',
    'bed2': 'indirect',
    'bed3': 'indirect',
    'bed4': 'indirect',
    'bathroom': 'indirect',
    'ensuite1': 'indirect',
    'ensuite2': 'indirect',
    'hall': 'indirect',
    'landing': 'indirect'
}
DEFAULT_EMITTER_KW = {
    'lounge': 1.4,
    'open_plan': 3.1,
    'utility': 0.6,
    'cloaks': 0.6,
    'bed1': 1.6,
    'bed2': 1.0,
    'bed3': 1.0,
    'bed4': 1.3,
    'bathroom': 0.39,
    'ensuite1': 0.39,
    'ensuite2': 0.39,
    'hall': 1.57,
    'landing': 1.1
}
DEFAULT_NUDGE_BUDGET = 2.5  # Lowered default to reduce over-boost

# HOUSE_CONFIG with defaults
HOUSE_CONFIG = {
    'rooms': DEFAULT_ROOMS,
    'facings': DEFAULT_FACINGS,
    'entities': DEFAULT_ENTITIES,
    'zone_sensor_map': DEFAULT_ZONE_SENSOR_MAP,
    'battery': DEFAULT_BATTERY,
    'grid': DEFAULT_GRID,
    'fallback_rates': DEFAULT_FALLBACK_RATES,
    'inverter': DEFAULT_INVERTER,
    'peak_loss': DEFAULT_PEAK_LOSS,
    'design_target': DEFAULT_DESIGN_TARGET,
    'peak_ext': DEFAULT_PEAK_EXT,
    'thermal_mass_per_m2': DEFAULT_THERMAL_MASS_PER_M2,
    'heat_up_tau_h': DEFAULT_HEAT_UP_TAU_H,
    'persistent_zones': DEFAULT_PERSISTENT_ZONES,
    'hp_flow_service': DEFAULT_HP_FLOW_SERVICE,
    'hp_hvac_service': DEFAULT_HP_HVAC_SERVICE,
    'room_control_mode': DEFAULT_ROOM_CONTROL_MODE,
    'emitter_kw': DEFAULT_EMITTER_KW,
    'nudge_budget': DEFAULT_NUDGE_BUDGET
}

# Safe override with type checks
def safe_override(key, default):
    if key in user_options:
        val = user_options[key]
        if isinstance(val, type(default)):
            HOUSE_CONFIG[key] = val
            logging.info(f"Overrode {key} from options.json")
        else:
            logging.error(f"Invalid type for {key} in options.json: expected {type(default)}, got {type(val)}. Using default.")

safe_override('rooms', DEFAULT_ROOMS)
safe_override('facings', DEFAULT_FACINGS)
safe_override('entities', DEFAULT_ENTITIES)
safe_override('zone_sensor_map', DEFAULT_ZONE_SENSOR_MAP)
safe_override('battery', DEFAULT_BATTERY)
safe_override('grid', DEFAULT_GRID)
safe_override('fallback_rates', DEFAULT_FALLBACK_RATES)
safe_override('inverter', DEFAULT_INVERTER)
safe_override('peak_loss', DEFAULT_PEAK_LOSS)
safe_override('design_target', DEFAULT_DESIGN_TARGET)
safe_override('peak_ext', DEFAULT_PEAK_EXT)
safe_override('thermal_mass_per_m2', DEFAULT_THERMAL_MASS_PER_M2)
safe_override('heat_up_tau_h', DEFAULT_HEAT_UP_TAU_H)
safe_override('persistent_zones', DEFAULT_PERSISTENT_ZONES)
safe_override('hp_flow_service', DEFAULT_HP_FLOW_SERVICE)
safe_override('hp_hvac_service', DEFAULT_HP_HVAC_SERVICE)
safe_override('room_control_mode', DEFAULT_ROOM_CONTROL_MODE)
safe_override('emitter_kw', DEFAULT_EMITTER_KW)
safe_override('nudge_budget', DEFAULT_NUDGE_BUDGET)

# Validate HOUSE_CONFIG after overrides
def validate_house_config(config):
    errors = []
    if not isinstance(config['rooms'], dict) or not all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in config['rooms'].items()):
        errors.append("Invalid 'rooms': must be dict with str keys and numeric values.")
    if not isinstance(config['facings'], dict) or not all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in config['facings'].items()):
        errors.append("Invalid 'facings': must be dict with str keys and numeric values.")
    if not isinstance(config['entities'], dict) or not all(isinstance(k, str) and isinstance(v, (str, list)) for k, v in config['entities'].items()):
        errors.append("Invalid 'entities': must be dict with str keys and str or list values.")
    if not isinstance(config['zone_sensor_map'], dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in config['zone_sensor_map'].items()):
        errors.append("Invalid 'zone_sensor_map': must be dict with str keys and str values.")
    if not isinstance(config['battery'], dict) or not all(k in DEFAULT_BATTERY for k in config['battery']):
        errors.append("Invalid 'battery': missing required keys.")
    if not isinstance(config['grid'], dict) or not all(k in DEFAULT_GRID for k in config['grid']):
        errors.append("Invalid 'grid': missing required keys.")
    if not isinstance(config['fallback_rates'], dict) or not all(k in DEFAULT_FALLBACK_RATES for k in config['fallback_rates']):
        errors.append("Invalid 'fallback_rates': missing required keys.")
    if not isinstance(config['inverter'], dict) or not all(k in DEFAULT_INVERTER for k in config['inverter']):
        errors.append("Invalid 'inverter': missing required keys.")
    if not isinstance(config['peak_loss'], (int, float)):
        errors.append("Invalid 'peak_loss': must be numeric.")
    if not isinstance(config['design_target'], (int, float)):
        errors.append("Invalid 'design_target': must be numeric.")
    if not isinstance(config['peak_ext'], (int, float)):
        errors.append("Invalid 'peak_ext': must be numeric.")
    if not isinstance(config['thermal_mass_per_m2'], (int, float)):
        errors.append("Invalid 'thermal_mass_per_m2': must be numeric.")
    if not isinstance(config['heat_up_tau_h'], (int, float)):
        errors.append("Invalid 'heat_up_tau_h': must be numeric.")
    if not isinstance(config['persistent_zones'], list) or not all(isinstance(z, str) for z in config['persistent_zones']):
        errors.append("Invalid 'persistent_zones': must be list of str.")
    if not isinstance(config['hp_flow_service'], dict) or not all(k in DEFAULT_HP_FLOW_SERVICE for k in config['hp_flow_service']):
        errors.append("Invalid 'hp_flow_service': missing required keys.")
    if not isinstance(config['hp_hvac_service'], dict) or not all(k in DEFAULT_HP_HVAC_SERVICE for k in config['hp_hvac_service']):
        errors.append("Invalid 'hp_hvac_service': missing required keys.")
    if not isinstance(config['room_control_mode'], dict) or not all(isinstance(k, str) and v in ['direct', 'indirect'] for k, v in config['room_control_mode'].items()):
        errors.append("Invalid 'room_control_mode': must be dict with str keys and 'direct' or 'indirect' values.")
    if not isinstance(config['emitter_kw'], dict) or not all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in config['emitter_kw'].items()):
        errors.append("Invalid 'emitter_kw': must be dict with str keys and numeric values.")
    if not isinstance(config['nudge_budget'], (int, float)):
        errors.append("Invalid 'nudge_budget': must be numeric.")

    if errors:
        for err in errors:
            logging.error(err)
        raise ValueError("HOUSE_CONFIG validation failed.")
    logging.info("HOUSE_CONFIG validated successfully.")

validate_house_config(HOUSE_CONFIG)

# Zone offsets
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

# Cycle detection constants
MIN_MODULATION_POWER = 0.20
LOW_POWER_MIN_IGNORE = 180
LOW_POWER_MAX_TOLERANCE = 1800
FLOW_TEMP_SPIKE_THRESHOLD = 1.5
POWER_SPIKE_THRESHOLD = 0.3
FLOW_TEMP_DROP_THRESHOLD = -1.5
COP_DROP_THRESHOLD = -0.3
COP_SPIKE_THRESHOLD = 0.3
DEMAND_DELTA_THRESHOLD = 1.5
LOSS_DELTA_THRESHOLD = 0.5
EXTENDED_RECOVERY_TIME = 300

# Globals for per-room nudges
room_nudge_hyst = {room: 0 for room in HOUSE_CONFIG['rooms']}
room_nudge_cooldown = {room: 0 for room in HOUSE_CONFIG['rooms']}
room_nudge_accum = {room: 0.0 for room in HOUSE_CONFIG['rooms']}

# Global for persisted rates
prev_all_rates = []

def parse_rates_array(rates_list, suppress_warning=False):
    if not rates_list or not isinstance(rates_list, list) or len(rates_list) == 0:
        if not suppress_warning:
            logging.warning("Rates list empty or invalid—using fallback rates.")
        return []
    try:
        return [(r['start'], r['end'], r['value_inc_vat']) for r in rates_list if 'start' in r and 'end' in r and 'value_inc_vat' in r]
    except Exception as e:
        logging.error(f"Rate parse error: {e} — using fallback rates.")
        return []

def get_current_rate(rates):
    now = datetime.now(timezone.utc)
    for start, end, price in rates:
        try:
            start_dt = datetime.fromisoformat(start)
            end_dt = datetime.fromisoformat(end)
            if start_dt <= now < end_dt:
                return price
        except ValueError as e:
            logging.warning(f"Invalid date in rates: {e} — skipping entry.")
    return HOUSE_CONFIG['fallback_rates']['standard']

def calc_solar_gain(config, production):
    return production * 0.5

def calc_room_loss(config, room, delta_temp, chill_factor=1.0, loss_coeff=0.0, sum_af=0.0):
    area = config['rooms'].get(room, 0)
    facing = config['facings'].get(room, 1.0)
    if sum_af == 0:
        return 0.0
    room_coeff = (area * facing / sum_af) * loss_coeff
    loss = room_coeff * max(0, delta_temp) * chill_factor
    return loss

def total_loss(config, ext_temp, room_targets, chill_factor=1.0, loss_coeff=0.0, sum_af=0.0):
    return sum(calc_room_loss(config, room, room_targets[room] - ext_temp, chill_factor, loss_coeff, sum_af) for room in config['rooms'])

def build_dfan_graph(config):
    G = nx.Graph()
    for room in config['rooms']:
        G.add_node(room, area=config['rooms'][room], facing=config['facings'][room])
    G.add_edges_from([('lounge', 'hall'), ('open_plan', 'utility')])
    return G

class SimpleQNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, action_dim))

    def forward(self, x):
        return self.fc(x)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = SimpleQNet(state_dim, action_dim)
        self.critic = SimpleQNet(state_dim, 1)

def train_rl(model, optimizer, episodes=500):
    for _ in range(episodes):
        states = torch.zeros(model.actor.fc[0].in_features)
        action = model.actor(states.unsqueeze(0))
        reward = random.uniform(-1, 1)
        value = model.critic(states.unsqueeze(0))
        loss = (reward - value).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logging.info("Initial RL training complete.")

# Globals for sim_step
action_counter = 0
prev_flow = 35.0
prev_mode = 'off'
prev_demand = 0.0
demand_history = deque(maxlen=5)
loss_history = deque(maxlen=5)
prod_history = deque(maxlen=5)
grid_history = deque(maxlen=5)
reward_history = deque(maxlen=100)
cop_history = deque(maxlen=10)  # For median COP fallback
low_power_start_time = None
cycle_type = None
cycle_start = None
pause_end = None
low_delta_persist = 0
undetected_count = 0
first_loop = True
prev_time = datetime.now()
blend_factor = 0.1
epsilon = 0.1
prev_hp_power = 1.0  # Initial value
prev_flow_temp = 35.0  # Initial value
prev_cop = 3.5  # Initial value
prev_actual_loss = 0.0  # Initial value

def sim_step(graph, states, config, model, optimizer, action_counter, prev_flow, prev_mode, prev_demand):
    global low_power_start_time, cycle_type, cycle_start, pause_end, low_delta_persist, undetected_count, first_loop, prev_time, blend_factor, epsilon, prev_hp_power, prev_flow_temp, prev_cop, prev_actual_loss
    try:
        current_time = datetime.now()
        dfan_control = fetch_ha_entity(config['entities']['dfan_control_toggle']) == 'on'
        soc = float(fetch_ha_entity(config['entities']['battery_soc']) or 0)
        ext_temp = float(fetch_ha_entity(config['entities']['outdoor_temp']) or 0)
        forecast_attrs = fetch_ha_entity(config['entities']['forecast_weather'], attr=True) or {}
        forecast_temp = forecast_attrs.get('temperature') or ext_temp
        forecast_min_temp = forecast_attrs.get('templow') or ext_temp
        wind_speed = forecast_attrs.get('wind_speed') or 0
        upcoming_cold = forecast_min_temp < 0
        upcoming_high_wind = wind_speed > 10
        chill_factor = 1 + (max(0, wind_speed - 5) / 20)
        solar_prod = float(fetch_ha_entity(config['entities']['solar_production']) or 0)
        prod_history.append(solar_prod)
        smoothed_prod = np.mean(prod_history)
        excess_solar = max(0, smoothed_prod - 1000)
        grid = float(fetch_ha_entity(config['entities']['grid_power']) or 0)
        grid_history.append(grid)
        smoothed_grid = np.mean(grid_history)
        hp_power = float(fetch_ha_entity(config['entities']['hp_output']) or 0) / 1000.0
        hp_energy = float(fetch_ha_entity(config['entities']['hp_energy_rate']) or 0) / 1000.0
        live_cop_raw = hp_power / hp_energy if hp_energy > 0 else 0
        cop_history.append(live_cop_raw)
        live_cop = np.median(cop_history) if np.median(cop_history) > 0 else 3.5
        current_flow_temp = float(fetch_ha_entity(config['entities']['hp_flow_temp']) or 35.0)
        delta_t = float(fetch_ha_entity(config['entities']['primary_diff']) or 0)
        hot_water_active = fetch_ha_entity(config['entities']['water_heater']) == 'heat'
        flow_min = float(fetch_ha_entity(config['entities']['flow_min_temp']) or 25.0)
        flow_max = float(fetch_ha_entity(config['entities']['flow_max_temp']) or 55.0)
        pid_target = float(fetch_ha_entity(config['entities']['pid_target_temperature']) or 21.0)
        current_day_rates = fetch_ha_entity(config['entities']['current_day_rates'], attr='rates') or []
        next_day_rates = fetch_ha_entity(config['entities']['next_day_rates'], attr='rates') or []
        all_rates = parse_rates_array(current_day_rates) + parse_rates_array(next_day_rates)
        current_rate = get_current_rate(all_rates) 

        room_targets = {}
        room_temps = {}
        open_fracs = []
        for room in config['rooms']:
            offset = ZONE_OFFSETS.get(room, 0)
            if room in config['persistent_zones']:
                room_targets[room] = 25.0
            else:
                room_targets[room] = pid_target + offset
            heating_entity = config['entities'].get(room + '_heating')
            temp = float(fetch_ha_entity(heating_entity) or 0)
            if temp > 0:
                room_temps[room] = temp
            else:
                sensor_key = config['zone_sensor_map'].get(room)
                room_temps[room] = float(fetch_ha_entity(config['entities'].get(sensor_key)) or 20.0)
            config_mode = config['room_control_mode'].get(room, 'indirect')
            if config_mode == 'direct':
                max_emitter = config['emitter_kw'].get(room, 1.0)
                open_frac = min(1.0, max(0.0, (room_targets[room] - room_temps[room]) / 5.0))
            else:
                open_frac = 1.0 if room_targets[room] > room_temps[room] else 0.0
            open_fracs.append(open_frac)
        avg_open_frac = np.mean(open_fracs) if open_fracs else 0.5

        sum_af = sum(config['rooms'][r] * config['facings'][r] for r in config['rooms'])
        loss_coeff = config['peak_loss'] / (config['design_target'] - config['peak_ext'])

        # Hybrid demand calc (reverted to v1.2.6 style with explicit hybrid)
        maintenance_loss = sum(calc_room_loss(config, room, room_temps[room] - ext_temp, chill_factor, loss_coeff, sum_af) for room in config['rooms'])
        deficit_loss = sum(calc_room_loss(config, room, room_targets[room] - room_temps[room], chill_factor, loss_coeff, sum_af) for room in config['rooms'])
        aggregate_heat_up = 0.0
        for room in config['rooms']:
            heat_up = max(0, (room_targets[room] - room_temps[room]) / config['heat_up_tau_h']) * config['rooms'][room] * config['thermal_mass_per_m2']
            aggregate_heat_up += heat_up

        # Dissipation nudges
        dissipation_fired = False
        total_disp = 0.0
        total_nudges = 0.0
        if avg_open_frac < 0.3 and (maintenance_loss + deficit_loss + aggregate_heat_up) > 2.0:
            dissipation_fired = True
            remaining_budget = config['nudge_budget']
            for room in config['rooms']:
                if room_nudge_cooldown[room] > 0:
                    room_nudge_cooldown[room] -= (current_time - prev_time).total_seconds()
                    continue
                config_mode = config['room_control_mode'].get(room, 'indirect')
                if config_mode == 'direct':
                    emitter_kw = config['emitter_kw'].get(room, 1.0)
                    nudge_kw = min(1.0, max(0.5, emitter_kw * (1 - open_fracs[list(config['rooms']).index(room)])))
                    if remaining_budget >= nudge_kw:
                        if room_nudge_accum[room] < 2.0:
                            room_nudge_accum[room] += nudge_kw
                            total_nudges += nudge_kw
                            remaining_budget -= nudge_kw
                            room_nudge_hyst[room] = 0.2
                            room_nudge_cooldown[room] = 300
                            total_disp += emitter_kw
                else:
                    nudge_kw = 0.5
                    if remaining_budget >= nudge_kw:
                        total_nudges += nudge_kw
                        remaining_budget -= nudge_kw

        total_demand = maintenance_loss + deficit_loss + aggregate_heat_up + total_nudges
        logging.info(f"Hybrid Demand Breakdown: Maintenance={maintenance_loss:.2f} kW, Deficit Loss={deficit_loss:.2f} kW, Heat-Up={aggregate_heat_up:.2f} kW, Nudges={total_nudges:.2f} kW, Total={total_demand:.2f} kW")

        demand_history.append(total_demand)
        smoothed_demand = np.mean(demand_history)
        loss_history.append(maintenance_loss + deficit_loss)  # For adjustment
        smoothed_loss = np.mean(loss_history)
        total_demand_adjusted = smoothed_demand * (1 + blend_factor * ((maintenance_loss + deficit_loss) / smoothed_loss - 1)) if smoothed_loss > 0 else smoothed_demand
        if total_demand_adjusted > 12.0:
            total_demand_adjusted *= 0.95  # Soft dampen to avoid hard cap
        demand_std = np.std(demand_history)

        if hot_water_active:
            optimal_mode = 'off'
            optimal_flow = 35.0
        else:
            optimal_mode = 'heat' if total_demand_adjusted > 0.5 else 'off'
            wc_cap = min(50, max(30, 50 - (ext_temp * 1.2)))
            optimal_flow = min(wc_cap, max(30, 30 + (total_demand_adjusted * 2.5)))
            if upcoming_cold or upcoming_high_wind:
                if current_rate < config['fallback_rates']['cheap']:
                    optimal_flow += 5.0
                optimal_flow = min(optimal_flow, flow_max)

        power_delta = hp_power - prev_hp_power
        flow_delta_actual = current_flow_temp - prev_flow_temp
        cop_delta = live_cop - prev_cop
        demand_delta = total_demand_adjusted - prev_demand
        loss_delta = (maintenance_loss + deficit_loss) - prev_actual_loss

        if first_loop:
            logging.info("Startup mode: Skipping cycle detection.")
            first_loop = False
        else:
            detected = False
            if flow_delta_actual <= FLOW_TEMP_DROP_THRESHOLD or cop_delta <= COP_DROP_THRESHOLD or demand_delta <= -DEMAND_DELTA_THRESHOLD or loss_delta <= -LOSS_DELTA_THRESHOLD:
                detected = True
                cycle_type = 'defrost'
            elif power_delta >= POWER_SPIKE_THRESHOLD or flow_delta_actual >= FLOW_TEMP_SPIKE_THRESHOLD or cop_delta >= COP_SPIKE_THRESHOLD or demand_delta >= DEMAND_DELTA_THRESHOLD or loss_delta >= LOSS_DELTA_THRESHOLD:
                detected = True
                cycle_type = 'oil_recovery'
            if detected:
                if cycle_start is None:
                    cycle_start = current_time
                logging.info(f"{cycle_type.capitalize()} cycle detected: Power delta {power_delta:.2f}kW / Flow delta {flow_delta_actual:.2f}°C / COP delta {cop_delta:.2f} / Demand delta {demand_delta:.2f} / Loss delta {loss_delta:.2f}")
                pause_end = current_time + timedelta(seconds=EXTENDED_RECOVERY_TIME)

        if hp_power < MIN_MODULATION_POWER and smoothed_demand > 0:
            if low_power_start_time is None:
                low_power_start_time = current_time
                logging.info("Low HP power detected: Monitoring for cycle patterns...")
            time_in_low = (current_time - low_power_start_time).total_seconds()
            if time_in_low > LOW_POWER_MIN_IGNORE and cycle_type:
                time_in_cycle = (current_time - cycle_start).total_seconds() if cycle_start else 0
                if time_in_cycle > EXTENDED_RECOVERY_TIME:
                    logging.info(f"Extended recovery pause: Type {cycle_type}, duration {time_in_cycle:.0f}s")
            if time_in_low > LOW_POWER_MAX_TOLERANCE:
                optimal_flow += 2.0
                optimal_flow = min(optimal_flow, flow_max)
                low_power_start_time = None
                logging.warning("Persistent low HP power without cycle patterns: Boosting demand")
        else:
            if low_power_start_time:
                time_in_low = (current_time - low_power_start_time).total_seconds()
                logging.info(f"HP power recovered: Ending monitor (cycle: {cycle_type or 'none'}, duration {time_in_low:.0f}s)")
            low_power_start_time = None

        if not first_loop and abs(demand_delta) >= DEMAND_DELTA_THRESHOLD and not cycle_type:
            logging.warning(f"Potential undetected cycle: large Δdemand {demand_delta:.2f} kW without patterns")
            undetected_count += 1

        if live_cop <= 0:
            logging.warning("Live COP <=0 outside cycle; using previous COP.")
            live_cop = prev_cop

        prev_hp_power = hp_power
        prev_flow_temp = current_flow_temp
        prev_cop = live_cop
        prev_actual_loss = maintenance_loss + deficit_loss
        prev_demand = total_demand_adjusted

        logging.info(f"Mode decision: optimal_mode='{optimal_mode}', total_demand={smoothed_demand:.2f} kW, "
                     f"ext_temp={ext_temp:.1f}°C, upcoming_cold={upcoming_cold}, upcoming_high_wind={upcoming_high_wind}, current_rate={current_rate:.3f} GBP/kWh, "
                     f"hot_water_active={hot_water_active}")

        states = torch.tensor([current_rate, soc, live_cop, optimal_flow, smoothed_demand, excess_solar, wind_speed, forecast_min_temp, smoothed_grid, delta_t, hp_power, chill_factor, demand_std, avg_open_frac], dtype=torch.float32)

        action = model.actor(states.unsqueeze(0))
        value = model.critic(states.unsqueeze(0))

        reward_adjust = 0
        if loss_history:
            prev_loss = loss_history[-1]
            if prev_loss > 100:
                reward_adjust -= 0.4
        if len(loss_history) >= 5 and sum(list(loss_history)[-5:]) / 5 < 1:
            reward_adjust += 0.4

        if pause_end and current_time >= pause_end:
            if len(demand_history) >= 5:
                recent_demand_std = np.std(list(demand_history)[-5:])
                if recent_demand_std < 0.5:
                    reward_adjust += 0.4
                    logging.info(f"Cycle-aware bonus: +0.4 for low demand_std {recent_demand_std:.2f} kW post-recovery (type: {cycle_type})")
                elif recent_demand_std > 1.0:
                    reward_adjust -= 0.3
                    logging.info(f"Cycle-aware penalty: -0.3 for high demand_std {recent_demand_std:.2f} kW post-recovery (type: {cycle_type})")
            pause_end = None
            cycle_type = None
            cycle_start = None

        reward = -0.8 * current_rate * total_demand_adjusted / live_cop if live_cop > 0 else 0.0
        reward += (live_cop - 3.0) * 0.5 - (abs(aggregate_heat_up) * 0.1)
        reward += reward_adjust

        if delta_t > 3.0:
            reward += 0.5
        if live_cop <= 0.5:
            reward -= 0.5

        if dissipation_fired:
            if total_disp > 5.0:
                reward += 0.3
            else:
                reward -= 0.2

        volatile = abs(demand_delta) > 1.0

        demand_penalty = 0.5 if abs(demand_delta) > 1.0 else 0
        flow_penalty = 0.3 if abs(flow_delta_actual) > 2.0 else 0
        ramp_penalty = 0.3 if abs(flow_delta_actual) > 1.5 else 0  # Assuming ramp_rate based on flow_delta
        dt_penalty = max(0, 3.0 - delta_t) * 2.0
        grid_penalty = abs(smoothed_grid / 1000.0) * current_rate * 0.2 if current_rate > 0.3 and smoothed_grid < -500 else 0
        export_bonus = (smoothed_grid / 1000.0) * config['fallback_rates']['export'] * 0.1 if smoothed_grid > 1000 else 0

        if volatile:
            demand_penalty *= 1.5
            flow_penalty *= 1.5
            ramp_penalty *= 1.5
            dt_penalty *= 1.5
            grid_penalty *= 1.5

        reward -= (demand_penalty + flow_penalty + ramp_penalty + dt_penalty + grid_penalty)
        reward += export_bonus

        if abs(demand_delta) < 0.5:
            reward += 0.3
        if abs(flow_delta_actual) < 0.5:
            reward += 0.2

        volatility_penalty = 0.0 if demand_std <= 0.5 else 0.2 * (demand_std - 0.5)
        reward -= volatility_penalty
        logging.info(f"DFAN RL: demand_std={demand_std:.2f} kW, volatility_penalty={volatility_penalty:.2f}")

        if ext_temp > 5 and optimal_flow > 40:
            reward -= 1.5
        if avg_open_frac < 0.5:
            penalty = 2.5 if any(config['room_control_mode'].get(r, 'indirect') == 'direct' for r in config['rooms']) else 2.0
            reward -= penalty

        if total_demand_adjusted > 12.0:
            reward -= 0.5  # New penalty for high demand

        original_cop = float(fetch_ha_entity(config['entities'].get('hp_cop')) or 0)
        if original_cop <= 0:
            logging.info("Skipping reward update due to COP <=0.")
            reward = 0.0
        td_error = reward - value.item()
        critic_loss = td_error ** 2

        norm_flow = (optimal_flow - 30) / 20
        flow_mse = (action[0][1] - torch.tensor(norm_flow)).pow(2)  # Fixed access
        actor_loss = flow_mse

        loss = critic_loss + (0.5 * actor_loss)

        if smoothed_demand > 50:
            loss = loss * 0.5

        optimizer.zero_grad()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.clamp_(-10, 10)
        optimizer.step()
        logging.info(f"RL update: Reward {reward:.2f}, Loss {loss.item():.4f}")

        reward_history.append(reward)
        loss_history.append(loss.item())  # Note: loss_history now for RL loss, but we use separate for demand loss

        epsilon = max(0.05, epsilon * 0.995)

        if len(reward_history) >= 10 and np.mean(list(reward_history)[-10:]) > 0:
            blend_factor = min(1.0, blend_factor + 0.01)

        flow_delta = abs(optimal_flow - prev_flow)
        demand_delta = abs(smoothed_demand - prev_demand)
        urgent = (optimal_mode != prev_mode) or (flow_delta > 2.0) or (demand_delta > 0.5) or (low_delta_persist >= 2) or (hp_power < 0.20)
        if dfan_control and (action_counter % 10 == 0 or urgent):
            for room in config['rooms']:
                entity_key = room + '_temp_set_hum'
                valve_entity = f'number.qsh_{room}_valve_target'
                config_mode = config['room_control_mode'].get(room, 'indirect')
                if entity_key in config['entities'] and (config_mode == 'indirect' or (config_mode == 'direct' and fetch_ha_entity(valve_entity) is None)):
                    temperature = room_targets[room]
                    data = {'entity_id': config['entities'][entity_key], 'temperature': temperature}
                    set_ha_service('climate', 'set_temperature', data)

            flow_data = {'device_id': config['hp_flow_service']['device_id'],
                         **config['hp_flow_service']['base_data'],
                         'weather_comp_min_temperature': flow_min,
                         'weather_comp_max_temperature': flow_max,
                         'fixed_flow_temperature': optimal_flow}
            set_ha_service(config['hp_flow_service']['domain'], config['hp_flow_service']['service'], flow_data)

            mode_data = {'device_id': config['hp_hvac_service']['device_id'], 'hvac_mode': optimal_mode}
            set_ha_service(config['hp_hvac_service']['domain'], config['hp_hvac_service']['service'], mode_data)

            if optimal_mode == 'heat':
                temp_data = {'device_id': config['hp_hvac_service']['device_id'], 'temperature': 23.0}
                set_ha_service('climate', 'set_temperature', temp_data)
                logging.info("DFAN setpoint adjust: Set HP temperature to static 23°C in 'heat' mode to override internal control.")

            logging.info(f"DFAN action triggered: { 'urgent' if urgent else 'scheduled' } - setting flow {optimal_flow:.1f}°C, mode {optimal_mode}.")
        else:
            logging.info(f"Data gather: Shadow would set flow {optimal_flow:.1f}°C and mode {optimal_mode}. (Action in {10 - (action_counter % 10)} loops)")

        clamped_demand = max(min(total_demand_adjusted, 15.0), 0.0)
        set_ha_service('input_number', 'set_value', {'entity_id': 'input_number.qsh_total_demand', 'value': clamped_demand})
        clamped_flow = max(25.0, min(55.0, optimal_flow))
        set_ha_service('input_number', 'set_value', {'entity_id': 'input_number.qsh_shadow_flow', 'value': clamped_flow})
        set_ha_service('input_select', 'select_option', {'entity_id': 'input_select.qsh_shadow_mode', 'option': optimal_mode})
        set_ha_service('input_select', 'select_option', {'entity_id': 'input_select.qsh_optimal_mode', 'option': optimal_mode})
        clamped_reward = max(min(reward, 100.0), -100.0)
        set_ha_service('input_number', 'set_value', {'entity_id': 'input_number.qsh_rl_reward', 'value': clamped_reward})
        clamped_loss = max(min(loss.item(), 2000.0), 0.0)
        set_ha_service('input_number', 'set_value', {'entity_id': 'input_number.qsh_rl_loss', 'value': clamped_loss})

        for room in config['rooms']:
            shadow_setpoint = room_targets[room]
            entity_id = f'input_number.qsh_shadow_{room}_setpoint'
            set_ha_service('input_number', 'set_value', {'entity_id': entity_id, 'value': shadow_setpoint})
            if not dfan_control:
                logging.info(f"Shadow: Would set {room} to {shadow_setpoint:.1f}°C")

        prev_time = current_time
        return action_counter + 1, optimal_flow, optimal_mode, total_demand_adjusted

    except Exception as e:
        logging.error(f"Sim step error: {e}")
        prev_time = current_time
        return action_counter + 1, prev_flow, prev_mode, prev_demand

graph = build_dfan_graph(HOUSE_CONFIG)
state_dim = 14
action_dim = 2
model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
states = torch.zeros(state_dim)

train_rl(model, optimizer)

def live_loop(graph, states, config, model, optimizer):
    global action_counter, prev_flow, prev_mode, prev_demand
    while True:
        action_counter, prev_flow, prev_mode, prev_demand = sim_step(graph, states, config, model, optimizer, action_counter, prev_flow, prev_mode, prev_demand)
        time.sleep(120)

live_loop(graph, states, HOUSE_CONFIG, model, optimizer)