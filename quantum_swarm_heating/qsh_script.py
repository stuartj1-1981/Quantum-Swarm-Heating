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
    if isinstance(entity_id, list):
        for eid in entity_id:
            data_single = data.copy()
            data_single['entity_id'] = eid
            try:
                r = requests.post(f"{HA_URL}/services/{domain}/{service}", headers=headers, json=data_single)
                r.raise_for_status()
            except Exception as e:
                logging.error(f"HA set error for {eid}: {e}")
    else:
        try:
            r = requests.post(f"{HA_URL}/services/{domain}/{service}", headers=headers, json=data)
            r.raise_for_status()
        except Exception as e:
            logging.error(f"HA set error for {entity_id or data.get('device_id')}: {e}")

# HOUSE_CONFIG
HOUSE_CONFIG = {
    'rooms': { 'lounge': 19.48, 'open_plan': 42.14, 'utility': 3.40, 'cloaks': 2.51,
        'bed1': 18.17, 'bed2': 13.59, 'bed3': 11.07, 'bed4': 9.79, 'bathroom': 6.02, 'ensuite1': 6.38, 'ensuite2': 3.71,
        'hall': 9.15, 'landing': 10.09 },
    'facings': { 'lounge': 0.2, 'open_plan': 1.0, 'utility': 0.5, 'cloaks': 0.5,
        'bed1': 0.2, 'bed2': 1.0, 'bed3': 0.5, 'bed4': 0.5, 'bathroom': 0.2, 'ensuite1': 0.5, 'ensuite2': 1.0,
        'hall': 0.2, 'landing': 0.2 },
    'entities': {
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
        'outdoor_temp': 'sensor.front_door_temperature_measurement',
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
        'hp_flow_temp': 'sensor.primary_flow_temperature'  
    },
    'zone_sensor_map': { 'hall': 'independent_sensor01', 'bed1': 'independent_sensor02', 'landing': 'independent_sensor03', 'open_plan': 'independent_sensor04',
        'utility': 'independent_sensor01', 'cloaks': 'independent_sensor01', 'bed2': 'independent_sensor02', 'bed3': 'independent_sensor03', 'bed4': 'independent_sensor03',
        'bathroom': 'independent_sensor03', 'ensuite1': 'independent_sensor02', 'ensuite2': 'independent_sensor03', 'lounge': 'independent_sensor01' },
    'battery': {'min_soc_reserve': 4.0, 'efficiency': 0.9, 'voltage': 51.0, 'max_rate': 3.0},
    'grid': {'nominal_voltage': 230.0, 'min_voltage': 200.0, 'max_voltage': 250.0},
    'fallback_rates': {'cheap': 0.1495, 'standard': 0.3048, 'peak': 0.4572, 'export': 0.15},
    'inverter': {'fallback_efficiency': 0.95},
    'peak_loss': 5.0,
    'design_target': 21.0,
    'peak_ext': -3.0,
    'thermal_mass_per_m2': 0.03,
    'heat_up_tau_h': 1.0,
    'persistent_zones': ['bathroom', 'ensuite1', 'ensuite2'],
    'hp_flow_service': {
        'domain': 'octopus_energy',
        'service': 'set_heat_pump_flow_temp_config',
        'device_id': 'b680894cd18521f7c706f1305b7333ea',
        'base_data': {
            'weather_comp_enabled': False
        }
    },
    'hp_hvac_service': {
        'domain': 'climate',
        'service': 'set_hvac_mode',
        'device_id': 'b680894cd18521f7c706f1305b7333ea'
    }
}

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
    'bathroom': 0.0,  # Persistent, so offset not added to 25
    'ensuite1': 0.0,
    'ensuite2': 0.0,
    'hall': 0.1,
    'landing': 0.4
}

# Cycle detection constants - updated thresholds for boosted sensitivity
MIN_MODULATION_POWER = 0.20
LOW_POWER_MIN_IGNORE = 180  # Reduced to 180s
LOW_POWER_MAX_TOLERANCE = 1800
FLOW_TEMP_SPIKE_THRESHOLD = 1.5  # Lowered
POWER_SPIKE_THRESHOLD = 0.3  # Lowered
FLOW_TEMP_DROP_THRESHOLD = -1.5  # Lowered abs for sensitivity
COP_DROP_THRESHOLD = -0.3  # Lowered
COP_SPIKE_THRESHOLD = 0.3  # Lowered
DEMAND_DELTA_THRESHOLD = 1.5  # Added for Δdemand
LOSS_DELTA_THRESHOLD = 0.5  # Added for loss spikes (kW)
EXTENDED_RECOVERY_TIME = 300  # For extended pauses

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

    def forward(self, x):
        action = self.actor(x)
        value = self.critic(x)
        return action, value

def train_rl(model, optimizer, episodes=500):
    for _ in range(episodes):
        # Simulate hypothetical state (random for variety)
        sim_state = torch.rand(13) * 10
        action, value = model(sim_state.unsqueeze(0))
        action = action.squeeze(0)
        
        # Simulate det logic approx (e.g., high demand → heat, high flow)
        sim_demand = sim_state[4].item()  # Index 4: smoothed_demand
        sim_mode = 1.0 if sim_demand > 5 else 0.0
        sim_flow = 45.0 if sim_demand > 5 else 35.0
        norm_flow = (sim_flow - 30) / 20
        
        # Simulate reward (good if matches sim)
        mode_match = (F.sigmoid(action[0]) > 0.5) == (sim_mode == 1.0)
        flow_err = abs(action[1].item() - norm_flow)
        sim_reward = 1.0 if mode_match and flow_err < 0.1 else -1.0
        
        td_error = sim_reward - value.item()
        critic_loss = td_error ** 2
        
        mode_log_prob = F.binary_cross_entropy_with_logits(action[0], torch.tensor(sim_mode))
        flow_mse = (action[1] - torch.tensor(norm_flow)).pow(2)
        actor_loss = mode_log_prob + flow_mse
        
        loss = critic_loss + 0.5 * actor_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logging.info("Initial RL training complete with simulated scenarios.")

# Globals for refinements
demand_history = deque(maxlen=5)  # Extended to 5
prod_history = deque(maxlen=5)
grid_history = deque(maxlen=5)
low_delta_persist = 0
low_power_start_time = None
prev_hp_power = 1.0  # Better init: assume moderate power
prev_flow_temp = 35.0  # Better init: typical flow
prev_cop = 3.5
cycle_type = None
cycle_start = None
pause_end = None  # New for fixed pause duration
action_counter = 0
prev_flow = 35.0
prev_mode = 'off'
prev_demand = 3.5  # Better init: fallback demand
prev_time = time.time()
prev_actual_loss = 0.0
reward_history = deque(maxlen=1000)
loss_history = deque(maxlen=1000)
pause_count = 0
undetected_count = 0
enable_plots = user_options.get('enable_plots', False)
first_loop = True  # New: Flag for startup
epsilon = 0.2  # Initial exploration rate
blend_factor = 0.5  # Initial RL blend factor

def shutdown_handler(sig, frame):
    mean_reward = sum(reward_history) / len(reward_history) if reward_history else 0
    mean_loss = sum(loss_history) / len(loss_history) if loss_history else 0
    demand_list = list(demand_history)
    demand_std = np.std(demand_list) if demand_list else 0
    logging.info(f"Shutdown summary: mean_reward={mean_reward:.2f}, mean_loss={mean_loss:.2f}, pause_count={pause_count}, demand_std={demand_std:.2f}, undetected_events={undetected_count}")
    if enable_plots and demand_list:
        import matplotlib.pyplot as plt  # Import here to avoid ModuleNotFoundError if not enabled
        plt.plot(demand_list)
        plt.title('Demand History')
        plt.xlabel('Steps')
        plt.ylabel('Demand (kW)')
        plt.savefig('/data/demand_hist.png')
        logging.info("Demand history plot saved to /data/demand_hist.png")
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

def sim_step(graph, states, config, model, optimizer, action_counter, prev_flow, prev_mode, prev_demand):
    global low_delta_persist, low_power_start_time, prev_hp_power, prev_flow_temp, prev_cop, cycle_type, demand_history, prod_history, grid_history, prev_time, cycle_start, prev_actual_loss, pause_count, undetected_count, first_loop, pause_end, prev_all_rates, epsilon, blend_factor
    try:
        current_time = time.time()
        time_delta = current_time - prev_time
        dfan_control = fetch_ha_entity(config['entities']['dfan_control_toggle']) == 'on'
        target_temp = float(fetch_ha_entity(config['entities']['pid_target_temperature']) or 21.0)
        logging.info(f"Using target_temp: {target_temp}°C from pid_target_temperature.")
        
        # Fetch hot_water_active early to decide on pause
        hot_water_active = fetch_ha_entity(config['entities']['water_heater']) == 'high_demand'
        
        # Compute room_targets (independent, for shadow logging only)
        room_targets = {room: target_temp + ZONE_OFFSETS.get(room, 0.0) for room in config['rooms']}
        for room in config['persistent_zones']:
            room_targets[room] = 25.0
        
        # Minimal fetching for log
        ext_temp = float(fetch_ha_entity(config['entities']['outdoor_temp']) or 0.0)
        current_day_rates = parse_rates_array(fetch_ha_entity(config['entities']['current_day_rates'], 'rates') or [], suppress_warning=(datetime.now(timezone.utc).hour < 16))
        current_rate = get_current_rate(current_day_rates)
        
        if hot_water_active:
            logging.info("Hot water active: Pausing all QSH processing (space heating optimizations, RL updates, cycle detection, and HA sets).")
            optimal_mode = 'heat'
            optimal_flow = prev_flow  # Retain previous to avoid jumps on resume
            total_demand_adjusted = 0.0
            
            # Minimal logging for mode decision (demand=0)
            logging.info(f"Mode decision: total_demand=0.00 kW, ext_temp={ext_temp:.1f}°C, upcoming_cold=False, upcoming_high_wind=False, current_rate={current_rate:.3f} GBP/kWh, hot_water_active=True")
            
            # Reset cycle/low-power states to avoid false positives on resume
            low_delta_persist = 0
            low_power_start_time = None
            cycle_type = None
            cycle_start = None
            pause_end = None
            
            # Set defaults for other variables to avoid undefined errors
            smoothed_demand = 0.0
            demand_std = 0.0
            chill_factor = 1.0
            smoothed_grid = 0.0
            forecast_min_temp = 0.0
            wind_speed = 0.0
            excess_solar = 0.0
            soc = 50.0
            live_cop = prev_cop
            delta_t = 3.0
            hp_power = float(fetch_ha_entity(config['entities']['hp_energy_rate']) or 0.0)
            upcoming_cold = False
            upcoming_high_wind = False
            actual_loss = 0.0
            sum_af = 0.0
            flow_min = float(fetch_ha_entity(config['entities']['flow_min_temp']) or 32.0)
            flow_max = float(fetch_ha_entity(config['entities']['flow_max_temp']) or 50.0)
            current_flow_temp = float(fetch_ha_entity(config['entities']['hp_flow_temp']) or 35.0)
            reward = 0.0  # Default, since no RL update
            loss = torch.tensor(0.0)  # Default
            heat_up_power = 0.0
            export_kw = 0.0
            
            # Skip to shadow sets and return
            clamped_demand = max(min(total_demand_adjusted, 15.0), 0.0)
            set_ha_service('input_number', 'set_value', {'entity_id': 'input_number.qsh_total_demand', 'value': clamped_demand})
            set_ha_service('input_number', 'set_value', {'entity_id': 'input_number.qsh_shadow_flow', 'value': optimal_flow})
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
        
        # Else: Normal processing
        # Fetch sensor temps
        sensor_temps = {
            'independent_sensor01': float(fetch_ha_entity(config['entities']['independent_sensor01']) or target_temp),
            'independent_sensor02': float(fetch_ha_entity(config['entities']['independent_sensor02']) or target_temp),
            'independent_sensor03': float(fetch_ha_entity(config['entities']['independent_sensor03']) or target_temp),
            'independent_sensor04': float(fetch_ha_entity(config['entities']['independent_sensor04']) or target_temp),
        }
        current_temps = {room: sensor_temps[config['zone_sensor_map'][room]] for room in config['rooms']}

        # Fetch forecast
        forecast = fetch_ha_entity(config['entities']['forecast_weather'], 'forecast') or []
        forecast_temps = [f['temperature'] for f in forecast if 'temperature' in f]
        forecast_wind_speeds = [f['wind_speed'] for f in forecast if 'wind_speed' in f]
        forecast_min_temp = min(forecast_temps or [ext_temp])
        wind_speed = float(fetch_ha_entity(config['entities']['forecast_weather'], 'wind_speed') or 0.0)
        upcoming_cold = any(t < 0 for t in forecast_temps)
        upcoming_high_wind = any(w > 15 for w in forecast_wind_speeds)
        chill_factor = 1.0 + (wind_speed / 10.0) * 0.1

        # Fetch solar, grid, soc, hp_output
        solar_production = float(fetch_ha_entity(config['entities']['solar_production']) or 0.0)
        grid_power = float(fetch_ha_entity(config['entities']['grid_power']) or 0.0)
        soc = float(fetch_ha_entity(config['entities']['battery_soc']) or 50.0)
        hp_output = float(fetch_ha_entity(config['entities']['hp_output']) or 0.0)
        excess_solar = max(0, solar_production - hp_output) / 1000.0  # Assume W to kW

        # Rates
        current_day_rates = parse_rates_array(fetch_ha_entity(config['entities']['current_day_rates'], 'rates') or [], suppress_warning=(datetime.now(timezone.utc).hour < 16))
        next_day_rates = parse_rates_array(fetch_ha_entity(config['entities']['next_day_rates'], 'rates') or [], suppress_warning=(datetime.now(timezone.utc).hour < 16))
        current_day_export = parse_rates_array(fetch_ha_entity(config['entities']['current_day_export_rates'], 'rates') or [], suppress_warning=(datetime.now(timezone.utc).hour < 16))
        next_day_export = parse_rates_array(fetch_ha_entity(config['entities']['next_day_export_rates'], 'rates') or [], suppress_warning=(datetime.now(timezone.utc).hour < 16))
        all_rates = current_day_rates + next_day_rates + current_day_export + next_day_export
        current_rate = get_current_rate(current_day_rates)

        # Loss calculation
        sum_af = sum(config['rooms'][room] * config['facings'][room] for room in config['rooms'])
        design_targets = {room: config['design_target'] for room in config['rooms']}
        base_loss = total_loss(config, config['peak_ext'], design_targets, 1.0, 1.0, sum_af)  # Fixed: use 1.0 for base
        loss_coeff = config['peak_loss'] / base_loss if base_loss > 0 else 0.0
        actual_loss = total_loss(config, ext_temp, room_targets, chill_factor, loss_coeff, sum_af)

        solar_gain = calc_solar_gain(config, solar_production / 1000.0)

        total_demand = max(0, actual_loss - solar_gain)
        heat_up_power = sum(config['thermal_mass_per_m2'] * config['rooms'][room] * max(0, room_targets[room] - current_temps[room]) / (config['heat_up_tau_h'] * 3600) for room in config['rooms'])
        total_demand_adjusted = total_demand + heat_up_power

        # Append to history and smooth
        demand_history.append(total_demand_adjusted)
        smoothed_demand = np.mean(demand_history)
        demand_std = np.std(demand_history) if len(demand_history) > 1 else 0.0

        prod_history.append(solar_production)
        grid_history.append(grid_power)
        smoothed_grid = np.mean(grid_history) if grid_history else 0.0
        export_kw = max(0, -smoothed_grid / 1000.0)  # Assuming negative grid is export

        delta_t = float(fetch_ha_entity(config['entities']['primary_diff']) or 3.0)
        hp_power = float(fetch_ha_entity(config['entities']['hp_energy_rate']) or 0.0)
        if delta_t < 1.0:
            low_delta_persist += 1
            if low_delta_persist >= 2:
                logging.info(f"DFAN ΔT safeguard: Persistent ΔT {delta_t:.1f}°C <1.0°C—preparing flow boost.")
        else:
            low_delta_persist = 0

        flow_min = float(fetch_ha_entity(config['entities']['flow_min_temp']) or 32.0)
        flow_max = float(fetch_ha_entity(config['entities']['flow_max_temp']) or 50.0)

        # Compute deterministic flow and mode
        det_flow = max(flow_min, min(flow_max, 35 + (smoothed_demand / config['peak_loss'] * (flow_max - 35))))
        wc_cap = min(50, max(30, 50 - (ext_temp * 1.2)))
        det_flow = min(det_flow, wc_cap)
        if upcoming_cold and current_rate < 0.15:
            det_flow += 5
            logging.info("Proactive heating would boost det_flow by 5°C due to forecast cold snap.")
        if upcoming_high_wind and current_rate < 0.15:
            det_flow += 3
            logging.info("Proactive heating would boost det_flow by 3°C due to forecast high wind.")
        det_flow = max(flow_min, min(flow_max, det_flow))

        det_mode = 'heat' if smoothed_demand > 1.5 or ext_temp < 5 or (upcoming_cold and current_rate < 0.15) or (upcoming_high_wind and current_rate < 0.15) else 'off'
        if excess_solar > 1 and det_mode == 'off':
            logging.info("Would have used 'auto' in old logic—defaulting to 'off' for QSH control.")

        # RL influence
        states = torch.tensor([current_rate, soc, prev_cop, prev_flow, smoothed_demand, excess_solar, wind_speed, forecast_min_temp, smoothed_grid, delta_t, hp_power, chill_factor, demand_std], dtype=torch.float32)
        action, value = model(states.unsqueeze(0))
        action = action.squeeze(0)

        mode_prob = F.sigmoid(action[0])
        if random.random() < epsilon:
            optimal_mode = 'heat' if random.random() > 0.5 else 'off'  # Explore
        else:
            optimal_mode = 'heat' if mode_prob > 0.5 else 'off'  # Greedy

        actor_flow = 30 + (F.sigmoid(action[1]) * 20)
        optimal_flow = (blend_factor * actor_flow.item()) + ((1 - blend_factor) * det_flow)

        # Apply proactive if mode heat
        if optimal_mode == 'heat':
            if upcoming_cold and current_rate < 0.15:
                optimal_flow += 5
            if upcoming_high_wind and current_rate < 0.15:
                optimal_flow += 3

        # Safeguards
        optimal_flow = min(optimal_flow, wc_cap)
        if low_delta_persist >= 2:
            optimal_flow += 2.0
        optimal_flow = max(flow_min, min(flow_max, optimal_flow))

        # Export safety override
        if soc > 80 and export_kw > 1 and current_rate > 0.3:
            optimal_mode = 'off'
            logging.info("Export optimized pause: high SOC and export during peak rate—overriding to 'off'")

        # Log overrides
        if optimal_mode != det_mode:
            logging.info(f"RL override: Changed mode from det '{det_mode}' to '{optimal_mode}' (prob: {mode_prob.item():.2f})")
        flow_diff = abs(optimal_flow - det_flow)
        if flow_diff > 2.0:
            logging.info(f"RL adjusted flow by {flow_diff:.1f}°C from det {det_flow:.1f}°C to {optimal_flow:.1f}°C")

        optimal_flow = round(optimal_flow, 1)
        flow_min = round(flow_min, 1)
        flow_max = round(flow_max, 1)
        logging.info(f"Rounded optimal_flow: {optimal_flow:.1f}°C, flow_min: {flow_min:.1f}°C, flow_max: {flow_max:.1f}°C")
        
        flow_delta = abs(optimal_flow - prev_flow)
        flow_ramp_rate = abs(flow_delta) / (time_delta / 60) if time_delta > 0 else 0
        
        current_flow_temp = float(fetch_ha_entity(config['entities']['hp_flow_temp']) or 35.0)
        cop_value = fetch_ha_entity(config['entities']['hp_cop'])
        if cop_value == 'unavailable':
            cop_value = fetch_ha_entity(config['entities']['hp_cop'])  # Redundancy
        if cop_value == 'unavailable':
            logging.warning("COP gap - potential undetected cycle")
            undetected_count += 1
            live_cop = prev_cop
        else:
            live_cop = float(cop_value) if cop_value else prev_cop

        power_delta = hp_power - prev_hp_power
        flow_delta_actual = current_flow_temp - prev_flow_temp  # Actual flow delta
        cop_delta = live_cop - prev_cop
        loss_delta = actual_loss - prev_actual_loss
        demand_delta = total_demand_adjusted - prev_demand  # Use adjusted

        # Check if in ongoing pause
        if pause_end and current_time < pause_end:
            time_remaining = pause_end - current_time
            logging.info(f"Ongoing cycle pause: Type {cycle_type}, remaining {time_remaining:.0f}s—pausing adjustments.")
            pause_count += 1
            # Update prev for accurate future deltas even during pause
            prev_hp_power = hp_power
            prev_flow_temp = current_flow_temp
            prev_cop = live_cop
            prev_actual_loss = actual_loss
            prev_demand = total_demand_adjusted
            prev_time = current_time
            return action_counter + 1, prev_flow, prev_mode, prev_demand

        # Cycle detection (skip on first_loop)
        if first_loop:
            logging.info("Startup mode: Skipping cycle detection to initialize prev values.")
            cycle_type = None
            cycle_start = None
            first_loop = False
        else:
            detected = False
            if (flow_delta_actual <= FLOW_TEMP_DROP_THRESHOLD or
                cop_delta <= COP_DROP_THRESHOLD or
                demand_delta <= -DEMAND_DELTA_THRESHOLD or
                loss_delta <= -LOSS_DELTA_THRESHOLD):
                detected = True
                cycle_type = 'defrost'
            elif (power_delta >= POWER_SPIKE_THRESHOLD or
                  flow_delta_actual >= FLOW_TEMP_SPIKE_THRESHOLD or
                  cop_delta >= COP_SPIKE_THRESHOLD or
                  demand_delta >= DEMAND_DELTA_THRESHOLD or
                  loss_delta >= LOSS_DELTA_THRESHOLD):
                detected = True
                cycle_type = 'oil_recovery'

            if detected:
                if cycle_start is None:
                    cycle_start = current_time
                logging.info(f"{cycle_type.capitalize()} cycle detected: Power delta {power_delta:.2f}kW / Flow delta {flow_delta_actual:.2f}°C / COP delta {cop_delta:.2f} / Demand delta {demand_delta:.2f} / Loss delta {loss_delta:.2f}")
                pause_end = current_time + EXTENDED_RECOVERY_TIME  # Set fixed pause on detection

        if hp_power < MIN_MODULATION_POWER and smoothed_demand > 0:
            if low_power_start_time is None:
                low_power_start_time = current_time
                logging.info("Low HP power detected: Monitoring for cycle patterns...")
            time_in_low = current_time - low_power_start_time
            if time_in_low > LOW_POWER_MIN_IGNORE and cycle_type:
                time_in_cycle = current_time - cycle_start if cycle_start else 0
                if time_in_cycle > EXTENDED_RECOVERY_TIME:
                    logging.info(f"Extended recovery pause: Type {cycle_type}, duration {time_in_cycle:.0f}s")
            if time_in_low > LOW_POWER_MAX_TOLERANCE:
                optimal_flow += 2.0
                optimal_flow = min(optimal_flow, flow_max)
                low_power_start_time = None
                logging.warning("Persistent low HP power without cycle patterns: Boosting demand")
        else:
            if low_power_start_time:
                time_in_low = current_time - low_power_start_time
                logging.info(f"HP power recovered: Ending monitor (cycle: {cycle_type or 'none'}, duration {time_in_low:.0f}s)")
            low_power_start_time = None
        
        if not first_loop and abs(demand_delta) >= DEMAND_DELTA_THRESHOLD and not cycle_type:
            logging.warning(f"Potential undetected cycle: large Δdemand {demand_delta:.2f} kW without patterns")
            undetected_count += 1
        
        if live_cop <= 0:
            logging.warning("Live COP <=0 outside cycle; using previous COP.")
            live_cop = prev_cop

        # Always update prev after detection/check
        prev_hp_power = hp_power
        prev_flow_temp = current_flow_temp
        prev_cop = live_cop
        prev_actual_loss = actual_loss
        prev_demand = total_demand_adjusted

        logging.info(f"Mode decision: optimal_mode='{optimal_mode}', total_demand={smoothed_demand:.2f} kW, "
                     f"ext_temp={ext_temp:.1f}°C, upcoming_cold={upcoming_cold}, upcoming_high_wind={upcoming_high_wind}, current_rate={current_rate:.3f} GBP/kWh, "
                     f"hot_water_active={hot_water_active}")

        # Reward adjustments from previous loss
        reward_adjust = 0
        if loss_history:
            prev_loss = loss_history[-1]
            if prev_loss > 100:
                reward_adjust -= 0.4
        if len(loss_history) >= 5 and sum(list(loss_history)[-5:]) / 5 < 1:
            reward_adjust += 0.4

        # Cycle-aware rewards: Trigger only on cycle completion
        if pause_end and current_time >= pause_end:
            if len(demand_history) >= 5:
                recent_demand_std = np.std(list(demand_history)[-5:])  # Last 5 for post-pause stability
                if recent_demand_std < 0.5:  # Stable recovery (tunable threshold)
                    reward_adjust += 0.4  # Bonus for smooth rebound
                    logging.info(f"Cycle-aware bonus: +0.4 for low demand_std {recent_demand_std:.2f} kW post-recovery (type: {cycle_type})")
                elif recent_demand_std > 1.0:  # Volatile recovery
                    reward_adjust -= 0.3  # Mild penalty
                    logging.info(f"Cycle-aware penalty: -0.3 for high demand_std {recent_demand_std:.2f} kW post-recovery (type: {cycle_type})")
            pause_end = None
            cycle_type = None
            cycle_start = None

        # Base reward with scaling
        reward = -0.8 * current_rate * total_demand_adjusted / live_cop
        reward += (live_cop - 3.0) * 0.5 - (abs(heat_up_power) * 0.1)
        reward += reward_adjust

        volatile = abs(demand_delta) > 1.0

        # Penalties and bonuses
        demand_penalty = 0.5 if abs(demand_delta) > 1.0 else 0
        flow_penalty = 0.3 if abs(flow_delta) > 2.0 else 0
        ramp_penalty = 0.3 if flow_ramp_rate > 1.5 else 0
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

        # Stability bonuses
        if abs(demand_delta) < 0.5:
            reward += 0.3
        if abs(flow_delta) < 0.5:
            reward += 0.2

        # DFAN RL Enhancement: Volatility penalty based on demand_std
        volatility_penalty = 0.0
        if demand_std > 0.5:
            volatility_penalty = 0.2 * (demand_std - 0.5)
        reward -= volatility_penalty
        logging.info(f"DFAN RL: demand_std={demand_std:.2f} kW, volatility_penalty={volatility_penalty:.2f}")

        # A2C loss
        td_error = reward - value.item()
        critic_loss = td_error ** 2

        optimal_mode_num = 1.0 if optimal_mode == 'heat' else 0.0
        mode_log_prob = F.binary_cross_entropy_with_logits(action[0], torch.tensor(optimal_mode_num))
        norm_flow = (optimal_flow - 30) / 20
        flow_mse = (action[1] - torch.tensor(norm_flow)).pow(2)
        actor_loss = mode_log_prob + flow_mse

        loss = critic_loss + (0.5 * actor_loss)

        # Loss scaling for anomalies
        if smoothed_demand > 50:
            loss = loss * 0.5

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.clamp_(-10, 10)

        optimizer.step()
        logging.info(f"RL update: Reward {reward:.2f}, Loss {loss.item():.4f}")

        reward_history.append(reward)
        loss_history.append(loss.item())

        # Decay epsilon
        epsilon = max(0.05, epsilon * 0.995)

        # Ramp blend_factor if recent rewards positive
        if len(reward_history) >= 10 and np.mean(list(reward_history)[-10:]) > 0:
            blend_factor = min(1.0, blend_factor + 0.01)

        flow_delta = abs(optimal_flow - prev_flow)
        demand_delta = abs(smoothed_demand - prev_demand)
        urgent = (optimal_mode != prev_mode) or (flow_delta > 2.0) or (demand_delta > 0.5) or (low_delta_persist >= 2) or (hp_power < 0.20)
        if dfan_control and (action_counter % 10 == 0 or urgent):
            for room in config['rooms']:
                entity_key = room + '_temp_set_hum'
                if entity_key in config['entities']:
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
        set_ha_service('input_number', 'set_value', {'entity_id': 'input_number.qsh_shadow_flow', 'value': optimal_flow})
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
state_dim = 13  # Updated for chill_factor + demand_std
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