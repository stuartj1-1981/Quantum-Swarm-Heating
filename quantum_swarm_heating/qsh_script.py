import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import random
from datetime import datetime, timedelta, timezone
import time
import logging
import os
import json
import requests
from collections import defaultdict, deque  # Added deque for smoothing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load user options (optional for now, as hardcoded; can re-enable for deployability)
try:
    with open('/data/options.json', 'r') as f:
        user_options = json.load(f)
except Exception as e:
    logging.warning(f"Failed to load options.json: {e}. Using defaults.")
    user_options = {}

logging.info(f"Loaded user_options: {user_options}")  # Debug

# HA API setup (use SUPERVISOR_TOKEN for internal add-on access)
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

# HOUSE_CONFIG (unchanged from prior—persistent_zones defined)
HOUSE_CONFIG = {
    'rooms': { 'lounge': 19.48, 'open_plan': 42.14, 'utility': 3.40, 'cloaks': 2.51,
        'bed1': 18.17, 'bed2': 13.59, 'bed3': 11.07, 'bed4': 9.79, 'bathroom': 6.02, 'ensuite1': 6.38, 'ensuite2': 3.71,
        'hall': 9.15, 'landing': 10.09 },
    'facings': { 'lounge': 0.2, 'open_plan': 1.0, 'utility': 0.5, 'cloaks': 0.5,
        'bed1': 0.2, 'bed2': 1.0, 'bed3': 0.5, 'bed4': 0.5, 'bathroom': 0.2, 'ensuite1': 0.5, 'ensuite2': 1.0,
        'hall': 0.2, 'landing': 0.2 },
    'entities': {
        'lounge_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va4240580352',
        'open_plan_temp_set_hum': ['climate.tado_smart_radiator_thermostat_va0349246464', 'climate.tado_smart_radiator_thermostat_va3553629184'],  # Dining and family room; add kitchen if separate (updated key)
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
        'pid_target_temperature': 'input_number.pid_target_temperature',  # Added for dynamic target_temp
        'grid_power': 'sensor.givtcp_ce2029g082_grid_power',  # Added for grid power integration
        'primary_diff': 'sensor.primary_diff',  # New for ΔT safeguards
        'hp_flow_temp': 'sensor.primary_flow_temperature'  
    },
    'zone_sensor_map': { 'hall': 'independent_sensor01', 'bed1': 'independent_sensor02', 'landing': 'independent_sensor03', 'open_plan': 'independent_sensor04',
        'utility': 'independent_sensor01', 'cloaks': 'independent_sensor01', 'bed2': 'independent_sensor02', 'bed3': 'independent_sensor03', 'bed4': 'independent_sensor03',
        'bathroom': 'independent_sensor03', 'ensuite1': 'independent_sensor02', 'ensuite2': 'independent_sensor03', 'lounge': 'independent_sensor01' },
    'battery': {'min_soc_reserve': 4.0, 'efficiency': 0.9, 'voltage': 51.0, 'max_rate': 3.0},
    'grid': {'nominal_voltage': 230.0, 'min_voltage': 200.0, 'max_voltage': 250.0},
    'fallback_rates': {'cheap': 0.1495, 'standard': 0.3048, 'peak': 0.4572, 'export': 0.15},
    'inverter': {'fallback_efficiency': 0.95},
    'peak_loss': 5.0,  # Reverted to 5.0 kW @ -3°C based on real world data from previous conversations
    'design_target': 21.0,  # Design internal temp for peak_loss calc
    'peak_ext': -3.0,       # Design external temp for peak_loss
    'thermal_mass_per_m2': 0.03,  # kWh/(m² °C), typical value based on residential buildings
    'heat_up_tau_h': 1.0,  # Desired heat-up time in hours for temperature deficits
    'persistent_zones': ['bathroom', 'ensuite1', 'ensuite2'],  # New for always-open balancing
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

# Cycle detection constants
MIN_MODULATION_POWER = 0.20  # kW
LOW_POWER_MIN_IGNORE = 300  # seconds (5 min initial ignore for cycles)
LOW_POWER_MAX_TOLERANCE = 1800  # seconds (30 min max before forced boost)
FLOW_TEMP_SPIKE_THRESHOLD = 2.0  # °C increase to detect oil recovery ramp
POWER_SPIKE_THRESHOLD = 0.5  # kW increase to detect ramp
FLOW_TEMP_DROP_THRESHOLD = -2.0  # °C decrease to detect defrost
COP_DROP_THRESHOLD = -0.5  # COP drop to confirm defrost (sudden efficiency dip)
COP_SPIKE_THRESHOLD = 0.5  # COP spike to confirm oil recovery (brief efficiency gain from ramp)

# Merge user options (optional for now, as hardcoded; can re-enable for deployability)

def parse_rates_array(rates_list):
    if not rates_list or not isinstance(rates_list, list) or len(rates_list) == 0:
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

def total_loss(config, ext_temp, target_temp=21.0, chill_factor=1.0, loss_coeff=0.0, sum_af=0.0):
    delta = target_temp - ext_temp
    return sum(calc_room_loss(config, room, delta, chill_factor, loss_coeff, sum_af) for room in config['rooms'])

def build_dfan_graph(config):
    G = nx.Graph()
    for room in config['rooms']:
        G.add_node(room, area=config['rooms'][room], facing=config['facings'][room])
    G.add_edges_from([('lounge', 'hall'), ('open_plan', 'utility')])  # Updated edge for renamed room
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

def train_rl(graph, states, config, model, optimizer, episodes=500):
    for _ in range(episodes):
        action = model.actor(states)
        reward = random.uniform(-1, 1)
        value = model.critic(states)
        loss = (reward - value).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logging.info("Initial RL training complete.")

def sim_step(graph, states, config, model, optimizer, action_counter, prev_flow, prev_mode, prev_demand):
    try:
        dfan_control = fetch_ha_entity(config['entities']['dfan_control_toggle']) == 'on'
        ext_temp = float(fetch_ha_entity(config['entities']['outdoor_temp']) or 0.0)
        wind_speed = float(fetch_ha_entity(config['entities']['forecast_weather'], 'wind_speed') or 0.0)
        chill_factor = 1.0
        target_temp = float(fetch_ha_entity(config['entities']['pid_target_temperature']) or 21.0) # Fetch from entity
        logging.info(f"Using target_temp: {target_temp}°C from pid_target_temperature.")
        delta = target_temp - ext_temp
        if wind_speed > 5:
            effective_temp = 13.12 + 0.6215 * ext_temp - 11.37 * wind_speed**0.16 + 0.3965 * ext_temp * wind_speed**0.16
            chill_delta = max(0, ext_temp - effective_temp)
            chill_factor = 1.0 + (chill_delta / max(1, delta))
        logging.info(f"Computed chill_factor: {chill_factor:.2f} based on wind {wind_speed} km/h")
        
        # Compute loss_coeff and sum_af here
        loss_coeff = config['peak_loss'] / (config['design_target'] - config['peak_ext']) if (config['design_target'] > config['peak_ext']) else 0.0
        sum_af = sum(config['rooms'][r] * config['facings'][r] for r in config['rooms'])
        logging.info(f"Computed loss_coeff: {loss_coeff:.3f} kW/°C, sum_af: {sum_af:.3f}")
        
        # Forecast with try-except for robustness
        try:
            forecast = fetch_ha_entity(config['entities']['forecast_weather'], 'forecast') or []
            forecast_temps = [f['temperature'] for f in forecast if 'temperature' in f and (datetime.fromisoformat(f['datetime']) - datetime.now()) < timedelta(hours=24)]
            forecast_min_temp = min(forecast_temps) if forecast_temps else ext_temp
            upcoming_cold = any(f['temperature'] < 5 
                                for f in forecast 
                                if 'temperature' in f 
                                and (datetime.fromisoformat(f['datetime']) - datetime.now()) < timedelta(hours=12))
        except Exception as e:
            logging.warning(f"Forecast fetch error: {e} — assuming no upcoming cold.")
            forecast_min_temp = ext_temp
            upcoming_cold = False

        # Fetch raw mode
        operation_mode = fetch_ha_entity(
            config['entities']['water_heater'],
            'operation_mode'
        )

        # Normalise against known valid modes
        valid_modes = {'electric', 'off', 'heat_pump', 'high_demand'}
        if operation_mode not in valid_modes:
            operation_mode = 'off'
            logging.warning("Unknown operation_mode—defaulting to 'off'.")

        # Detect high-demand mode
        hot_water_active = operation_mode == 'high_demand'

        if hot_water_active:
            logging.info("Hot water cycle active—pausing space heating sets.")
            return action_counter + 1, prev_flow, prev_mode, prev_demand

        # Build zone groups
        sensor_to_rooms = defaultdict(list)
        for room, sensor_key in config['zone_sensor_map'].items():
            sensor_to_rooms[sensor_key].append(room)

        # Fetch zone temps and compute offsets
        zone_temps = {}
        zone_offsets = {}  # For setpoints, per room
        for sensor_key, rooms_list in sensor_to_rooms.items():
            sensor_entity = config['entities'].get(sensor_key)
            if sensor_entity:
                zone_temp = float(fetch_ha_entity(sensor_entity) or target_temp)
                zone_temps[sensor_key] = zone_temp
                for room in rooms_list:
                    zone_offsets[room] = target_temp - zone_temp  # Same offset for all rooms in zone

        # Compute actual_loss and heat_up_power
        actual_loss = 0.0
        heat_up_power = 0.0
        for sensor_key, rooms_list in sensor_to_rooms.items():
            zone_temp = zone_temps[sensor_key]
            zone_af = sum(config['rooms'][r] * config['facings'][r] for r in rooms_list)
            zone_loss = total_loss(config, ext_temp, target_temp, chill_factor, loss_coeff, zone_af)
            actual_loss += zone_loss
            deficit = max(0, target_temp - zone_temp)
            zone_mass = sum(config['rooms'][r] for r in rooms_list) * config['thermal_mass_per_m2']
            heat_up = (deficit * zone_mass) / config['heat_up_tau_h']
            heat_up_power += heat_up

        # Rates with fallback
        current_day_rates_list = fetch_ha_entity(config['entities']['current_day_rates']) or []
        next_day_rates_list = fetch_ha_entity(config['entities']['next_day_rates']) or []
        current_day_parsed = parse_rates_array(current_day_rates_list)
        next_day_parsed = parse_rates_array(next_day_rates_list)

        all_rates = current_day_parsed + next_day_parsed
        current_rate = get_current_rate(all_rates)
        next_cheap = min(price for _, _, price in all_rates)  if all_rates else config['fallback_rates']['cheap']

        production = float(fetch_ha_entity(config['entities']['solar_production']) or 0)
        global prod_history
        prod_history.append(production)
        smoothed_prod = sum(prod_history) / len(prod_history)
        solar_gain = calc_solar_gain(config, smoothed_prod)
        excess_solar = max(0, smoothed_prod)
        total_demand = actual_loss + heat_up_power - solar_gain

        soc = float(fetch_ha_entity(config['entities']['battery_soc']) or 50.0)
        charge_rate = 0.0
        discharge_rate = 0.0
        
        # Fetch grid power (positive: export, negative: import)
        grid_power = float(fetch_ha_entity(config['entities']['grid_power']) or 0.0)
        global grid_history
        grid_history.append(grid_power)
        smoothed_grid = sum(grid_history) / len(grid_history)
        logging.info(f"Fetched grid_power: {grid_power:.2f} W, Smoothed: {smoothed_grid:.2f} W")
        logging.info(f"Fetched solar_production: {production:.2f} kW, Smoothed: {smoothed_prod:.2f} kW")

        # New: Demand smoothing
        global demand_history
        demand_history.append(total_demand)
        smoothed_demand = sum(demand_history) / len(demand_history)

        # Revised: total_demand_adjusted
        import_kw = max(0, -smoothed_grid / 1000.0)  # If negative grid (import), positive kW
        export_kw = max(0, smoothed_grid / 1000.0)  # If positive, export kW
        total_demand_adjusted = max(0, smoothed_demand - excess_solar - export_kw + import_kw)

        # Guard against large swings
        if abs(total_demand_adjusted - prev_demand) > 2.0:
            logging.warning(f"Input anomaly detected: Demand swing {total_demand_adjusted:.2f} from {prev_demand:.2f} kW—using fallback.")
            total_demand_adjusted = (total_demand_adjusted + prev_demand) / 2.0  # Average or prev

        # Updated: DFAN ΔT safeguard <1.0°C with persistence
        global low_delta_persist
        delta_t = float(fetch_ha_entity(config['entities']['primary_diff']) or 3.0)
        hp_power = float(fetch_ha_entity(config['entities']['hp_energy_rate']) or 0.0)  # kW
        if delta_t < 1.0:
            low_delta_persist += 1
            if low_delta_persist >= 2:
                logging.info(f"DFAN ΔT safeguard: Persistent ΔT {delta_t:.1f}°C <1.0°C—preparing flow boost.")
        else:
            low_delta_persist = 0

        flow_min = float(fetch_ha_entity(config['entities']['flow_min_temp']) or 32.0)
        flow_max = float(fetch_ha_entity(config['entities']['flow_max_temp']) or 50.0)
        optimal_flow = max(flow_min, min(flow_max, 35 + (smoothed_demand / config['peak_loss'] * (flow_max - 35))))
        
        # DFAN WC cap (low limit 30°C as requested)
        wc_cap = min(45, max(30, 50 - (ext_temp * 1.2)))  # Tune multiplier
        optimal_flow = min(optimal_flow, wc_cap)
        logging.info(f"DFAN WC applied: Capped flow to {optimal_flow:.1f}°C for ext_temp {ext_temp:.1f}°C")
        
        # Apply ΔT boost if persistent (after capping)
        if low_delta_persist >= 2:
            optimal_flow += 2.0
            optimal_flow = min(optimal_flow, flow_max)
            logging.info(f"DFAN ΔT safeguard: Boosted flow by 2°C (persistent ΔT: {delta_t:.1f}°C)")

        # Fix Octopus GraphQL serializer validation (KT-CT-4321) – too many float digits
        optimal_flow = round(optimal_flow, 1)
        flow_min = round(flow_min, 1)
        flow_max = round(flow_max, 1)
        logging.info(f"Rounded optimal_flow: {optimal_flow:.1f}°C, flow_min: {flow_min:.1f}°C, flow_max: {flow_max:.1f}°C")
        
        # MODIFIED: Binary mode choice only ('heat' or 'off') to avoid HP internal schedule
        optimal_mode = 'heat'  # Default assume heat
        if smoothed_demand > 1.5 or ext_temp < 5 or (upcoming_cold and current_rate < 0.15):
            optimal_mode = 'heat'  # Prioritize heat for demand/cold/cheap proactive scenarios
            if upcoming_cold and current_rate < 0.15:
                optimal_flow += 5
                logging.info("Proactive heating enabled due to forecast cold snap.")
        else:
            optimal_mode = 'off'  # Default to off (e.g., excess solar or low demand)
            if excess_solar > 1:
                logging.info("Would have used 'auto' in old logic—defaulting to 'off' for QSH control.")

        # Fetch current flow temp and COP for cycle detection
        current_flow_temp = float(fetch_ha_entity(config['entities']['hp_flow_temp']) or 35.0)
        cop_value = fetch_ha_entity(config['entities']['hp_cop'])
        live_cop = float(cop_value) if cop_value and cop_value != 'unavailable' else prev_cop

        # Min Modulation with Cycle Detection (Oil Recovery/Defrost) - replaces old min mod check
        global low_power_start_time, prev_hp_power, prev_flow_temp, prev_cop, cycle_type
        current_time = time.time()
        power_delta = hp_power - prev_hp_power
        flow_delta = current_flow_temp - prev_flow_temp
        cop_delta = live_cop - prev_cop
        
        if hp_power < MIN_MODULATION_POWER and smoothed_demand > 0:
            if low_power_start_time is None:
                low_power_start_time = current_time
                logging.info("Low HP power detected: Monitoring for cycle patterns...")
            
            time_in_low = current_time - low_power_start_time
            
            if time_in_low > LOW_POWER_MIN_IGNORE:
                if flow_delta <= FLOW_TEMP_DROP_THRESHOLD or cop_delta <= COP_DROP_THRESHOLD:
                    cycle_type = 'defrost'
                    logging.info(f"Defrost cycle detected: Flow delta {flow_delta:.2f}°C / COP delta {cop_delta:.2f}. Pausing...")
                elif power_delta >= POWER_SPIKE_THRESHOLD or flow_delta >= FLOW_TEMP_SPIKE_THRESHOLD or cop_delta >= COP_SPIKE_THRESHOLD:
                    cycle_type = 'oil_recovery'
                    logging.info(f"Oil recovery cycle detected: Power delta {power_delta:.2f}kW / Flow delta {flow_delta:.2f}°C / COP delta {cop_delta:.2f}. Pausing...")
                
                if cycle_type:
                    # NEW: Pause like hot water
                    logging.info(f"Cycle pause: Type {cycle_type}, duration {time_in_low:.0f}s—pausing adjustments.")
                    return action_counter + 1, prev_flow, prev_mode, prev_demand  # Skip updates/RL/sets
                
                elif time_in_low > LOW_POWER_MAX_TOLERANCE:
                    optimal_flow += 2.0  # Boost flow as demand boost
                    optimal_flow = min(optimal_flow, flow_max)
                    low_power_start_time = None
                    cycle_type = None
                    logging.warning("Persistent low HP power without cycle patterns: Boosting demand")
        else:
            if low_power_start_time:
                logging.info(f"HP power recovered: Ending monitor (cycle: {cycle_type or 'none'}, duration {time_in_low:.0f}s)")
            low_power_start_time = None
            cycle_type = None
        
        # NEW: COP check outside cycles—use prev if <=0
        if live_cop <= 0:
            logging.warning("Live COP <=0 outside cycle; using previous COP.")
            live_cop = prev_cop

        # Update prev for next loop
        prev_hp_power = hp_power
        prev_flow_temp = current_flow_temp
        prev_cop = live_cop

        # Debug logging for mode decision
        logging.info(f"Mode decision: optimal_mode='{optimal_mode}', total_demand={smoothed_demand:.2f} kW, "
                     f"ext_temp={ext_temp:.1f}°C, upcoming_cold={upcoming_cold}, current_rate={current_rate:.3f} GBP/kWh, "
                     f"hot_water_active={hot_water_active}")

        # Expand states to include grid_power, delta_t, hp_power (state_dim now 11)
        states = torch.tensor([current_rate, soc, live_cop, optimal_flow, smoothed_demand, excess_solar, wind_speed, forecast_min_temp, smoothed_grid, delta_t, hp_power], dtype=torch.float32)

        action = model.actor(states.unsqueeze(0))
        reward = -current_rate * total_demand_adjusted / live_cop
        reward += (live_cop - 3.0) * 0.5 - (abs(heat_up_power) * 0.1)
        
        # RL reward tweak for grid power (penalize imports during peaks, bonus for exports)
        if current_rate > 0.3 and smoothed_grid < -500:  # Import >0.5 kW
            reward -= abs(smoothed_grid / 1000.0) * current_rate * 0.2  # Cost hit for unoffset imports
        elif smoothed_grid > 1000:  # Export >1 kW
            reward += (smoothed_grid / 1000.0) * config['fallback_rates']['export'] * 0.1  # Export bonus
        
        # New: ΔT penalty in reward
        reward -= max(0, 3.0 - delta_t) * 2.0  # Penalty if <3°C

        # New: Penalize rapid flow changes
        if abs(optimal_flow - prev_flow) > 2.0:
            reward -= 0.3

        # Amped: Penalize rapid demand changes
        if abs(total_demand_adjusted - prev_demand) > 1.0:
            reward -= 0.5

        value = model.critic(states.unsqueeze(0))
        loss = (reward - value).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logging.info(f"RL update: Reward {reward:.2f}, Loss {loss.item():.4f}")

        # Conditional actions: Every 10 loops or urgent
        flow_delta = abs(optimal_flow - prev_flow)
        demand_delta = abs(smoothed_demand - prev_demand)
        urgent = (optimal_mode != prev_mode) or (flow_delta > 2.0) or (demand_delta > 0.5) or (low_delta_persist >= 2) or (hp_power < 0.20)
        if dfan_control and (action_counter % 10 == 0 or urgent):
            for room in config['rooms']:
                entity_key = room + '_temp_set_hum'
                if entity_key in config['entities']:
                    temperature = 25.0 if room in config['persistent_zones'] else target_temp + zone_offsets.get(room, 0)
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
            
            # Updated: Set static 23°C only when mode='heat' (HP auto-handles 'off' to 7°C)
            if optimal_mode == 'heat':
                temp_data = {'device_id': config['hp_hvac_service']['device_id'], 'temperature': 23.0}
                set_ha_service('climate', 'set_temperature', temp_data)
                logging.info("DFAN setpoint adjust: Set HP temperature to static 23°C in 'heat' mode to override internal control.")

            logging.info(f"DFAN action triggered: { 'urgent' if urgent else 'scheduled' } - setting flow {optimal_flow:.1f}°C, mode {optimal_mode}.")
        else:
            logging.info(f"Data gather: Shadow would set flow {optimal_flow:.1f}°C and mode {optimal_mode}. (Action in {10 - (action_counter % 10)} loops)")

        # Shadow/preview entities (always, for dashboard consistency) with try-except for robustness
        try:
            clamped_demand = max(min(total_demand_adjusted, 10.0), 0.0)
            set_ha_service('input_number', 'set_value', {'entity_id': 'input_number.qsh_total_demand', 'value': clamped_demand})
        except Exception as e:
            logging.warning(f"Shadow set failed for qsh_total_demand: {e}")

        try:
            set_ha_service('input_number', 'set_value', {'entity_id': 'input_number.qsh_shadow_flow', 'value': optimal_flow})
        except Exception as e:
            logging.warning(f"Shadow set failed for qsh_shadow_flow: {e}")

        try:
            set_ha_service('input_select', 'select_option', {'entity_id': 'input_select.qsh_shadow_mode', 'option': optimal_mode})
        except Exception as e:
            logging.warning(f"Shadow set failed for qsh_shadow_mode: {e}")

        try:
            set_ha_service('input_select', 'select_option', {'entity_id': 'input_select.qsh_optimal_mode', 'option': optimal_mode})
        except Exception as e:
            logging.warning(f"Shadow set failed for qsh_optimal_mode: {e}")

        # RL metrics (with clamping to match entity min/max)
        try:
            clamped_reward = max(min(reward, 100.0), -100.0)
            set_ha_service('input_number', 'set_value', {'entity_id': 'input_number.qsh_rl_reward', 'value': clamped_reward})
        except Exception as e:
            logging.warning(f"Shadow set failed for qsh_rl_reward: {e}")

        try:
            clamped_loss = max(min(loss.item(), 2000.0), 0.0)
            set_ha_service('input_number', 'set_value', {'entity_id': 'input_number.qsh_rl_loss', 'value': clamped_loss})
        except Exception as e:
            logging.warning(f"Shadow set failed for qsh_rl_loss: {e}")

        # Per-room shadow setpoints (with static override for persistent)
        for room in config['rooms']:
            try:
                shadow_setpoint = 25.0 if room in config['persistent_zones'] else target_temp + zone_offsets.get(room, 0)
                entity_id = f'input_number.qsh_shadow_{room}_setpoint'
                set_ha_service('input_number', 'set_value', {'entity_id': entity_id, 'value': shadow_setpoint})
                if not dfan_control:
                    logging.info(f"Shadow: Would set {room} to {shadow_setpoint:.1f}°C")  # Optional extra log for debug
            except Exception as e:
                logging.warning(f"Shadow set failed for qsh_shadow_{room}_setpoint: {e}")

        return action_counter + 1, optimal_flow, optimal_mode, total_demand_adjusted

    except Exception as e:
        logging.error(f"Sim step error: {e}")
        return action_counter + 1, prev_flow, prev_mode, prev_demand

graph = build_dfan_graph(HOUSE_CONFIG)
state_dim = 11  # Updated to 11 for delta_t and hp_power addition
action_dim = 2
model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
states = torch.zeros(state_dim)

train_rl(graph, states, HOUSE_CONFIG, model, optimizer)

# Initialize globals for refinements
demand_history = deque(maxlen=3)
low_delta_persist = 0
low_power_start_time = None
prev_hp_power = 0.0
prev_flow_temp = 0.0
prev_cop = 3.5  # New: Track previous COP
cycle_type = None
action_counter = 0
prev_flow = 35.0
prev_mode = 'off'
prev_demand = 0.0
prod_history = deque(maxlen=3)
grid_history = deque(maxlen=3)

def live_loop(graph, states, config, model, optimizer):
    global action_counter, prev_flow, prev_mode, prev_demand
    while True:
        action_counter, prev_flow, prev_mode, prev_demand = sim_step(graph, states, config, model, optimizer, action_counter, prev_flow, prev_mode, prev_demand)
        time.sleep(120)  # 2min gather cycles

live_loop(graph, states, HOUSE_CONFIG, model, optimizer)