import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import random
from datetime import datetime, timedelta
import time
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check if 'hass' is available (HA injects it in add-ons/components)
try:
    hass
except NameError:
    logging.critical("'hass' not defined! Running in fallback mode with defaults only. Ensure script is in a HA add-on environment.")
    class MockStates:
        def get(self, eid):
            return None
    class MockServices:
        def call(self, *args, **kwargs):
            logging.warning("Mock service call—no action taken.")
    class MockHass:
        states = MockStates()
        services = MockServices()
    hass = MockHass()

def fetch_ha_entity(entity_id, attr=None):
    state = hass.states.get(entity_id)
    if state is None:
        logging.warning(f"Entity {entity_id} not found—using default.")
        return None
    if attr:
        return state.attributes.get(attr)
    return state.state

def set_ha_service(domain, service, data):
    entity_id = data.get('entity_id')
    if isinstance(entity_id, list):
        for eid in entity_id:
            data_single = data.copy()
            data_single['entity_id'] = eid
            hass.services.call(domain, service, data_single, blocking=True)
    else:
        hass.services.call(domain, service, data, blocking=True)

# Default config (unchanged from prior DFAN iterations)
HOUSE_CONFIG = {
    'rooms': { 'lounge': 19.48, 'open_plan_ground': 42.14, 'utility': 3.40, 'cloaks': 2.51,
        'bed1': 18.17, 'bed2': 13.59, 'bed3': 11.07, 'bed4': 9.79, 'bathroom': 6.02, 'ensuite1': 6.38, 'ensuite2': 3.71,
        'hall': 9.15, 'landing': 10.09 },
    'facings': { 'lounge': 0.2, 'open_plan_ground': 1.0, 'utility': 0.5, 'cloaks': 0.5,
        'bed1': 0.2, 'bed2': 1.0, 'bed3': 0.5, 'bed4': 0.5, 'bathroom': 0.2, 'ensuite1': 0.5, 'ensuite2': 1.0,
        'hall': 0.2, 'landing': 0.2 },
    'entities': {
        'lounge_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va4240580352',
        'open_plan_ground_temp_set_hum': ['climate.tado_smart_radiator_thermostat_va0349246464', 'climate.tado_smart_radiator_thermostat_va3553629184'],
        # Add other Tado rooms here if not overridden (e.g., 'bed1_temp_set_hum': 'climate.tado_...')
        'independent_sensor01': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor01_temperature',
        'independent_sensor02': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor02_temperature',
        'independent_sensor03': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor03_temperature',
        'independent_sensor04': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor04_temperature',
        'battery_soc': 'sensor.givtcp_ce2029g082_soc',
        'battery_design_capacity_ah': 'sensor.givtcp_dx2327m548_battery_design_capacity',  # Note: Prefix mismatch; unify if possible
        'battery_remaining_capacity_ah': 'sensor.givtcp_dx2327m548_battery_remaining_capacity',
        'battery_power': 'sensor.givtcp_ce2029g082_battery_power',
        'battery_voltage': 'sensor.givtcp_ba2027g052_battery_voltage_2',
        'ac_charge_power': 'sensor.givtcp_ce2029g082_ac_charge_power',
        'battery_to_grid': 'sensor.givtcp_ce2029g082_battery_to_grid',
        'battery_to_house': 'sensor.givtcp_ce2029g082_battery_to_house',
        'grid_voltage_2': 'sensor.givtcp_ce2029g082_grid_voltage_2',
        'grid_power': 'sensor.givtcp_ce2029g082_grid_power',
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
        'hp_water_tonight': 'input_boolean.hp_chosen_for_tonight',
        'water_heater': 'water_heater.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31',
        'flow_min_temp': 'input_number.flow_min_temperature',
        'flow_max_temp': 'input_number.flow_max_temperature',
        'hp_cop': 'sensor.live_cop_calc',
        'dfan_control_toggle': 'input_boolean.dfan_control'
    },
    'zone_sensor_map': { 'hall': 'independent_sensor01', 'bed1': 'independent_sensor02', 'landing': 'independent_sensor03', 'open_plan_ground': 'independent_sensor04',
        'utility': 'independent_sensor01', 'cloaks': 'independent_sensor01', 'bed2': 'independent_sensor02', 'bed3': 'independent_sensor03', 'bed4': 'independent_sensor03',
        'bathroom': 'independent_sensor03', 'ensuite1': 'independent_sensor02', 'ensuite2': 'independent_sensor03', 'lounge': 'independent_sensor01' },
    'hot_water': {'load_kw': 2.5, 'ext_threshold': 3.0, 'cycle_start_hour': 0, 'cycle_end_hour': 6, 'tank_low_threshold': 40.0},  # Added default threshold
    'battery': {'min_soc_reserve': 4.0, 'efficiency': 0.9, 'voltage': 51.0, 'max_rate': 3.0},
    'grid': {'nominal_voltage': 230.0, 'min_voltage': 200.0, 'max_voltage': 250.0},
    'fallback_rates': {'cheap': 0.1495, 'standard': 0.3048, 'peak': 0.4572, 'export': 0.15},
    'inverter': {'fallback_efficiency': 0.95},
    'peak_loss': 10.0,  # Added default (adjust based on your peak heat loss in kW)
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

# Load user options from add-on (fallback to /data/options.json if not in HA)
try:
    with open('/data/options.json', 'r') as f:
        options = json.load(f)
except Exception as e:
    logging.warning(f"Failed to load options.json: {e}. Using defaults.")
    options = {}

# Merge user options (expand for other sections as needed, e.g., if options have 'battery_entities')
if 'tado_rooms' in options and isinstance(options['tado_rooms'], list):
    HOUSE_CONFIG['entities'].update({item['room'] + '_temp_set_hum': item['entity'] for item in options['tado_rooms'] if isinstance(item, dict) and 'room' in item and 'entity' in item})
if 'independent_sensors' in options and isinstance(options['independent_sensors'], list):
    HOUSE_CONFIG['entities'].update({item['sensor'] for item in options['independent_sensors'] if isinstance(item, dict) and 'sensor' in item})
if 'battery_entities' in options and isinstance(options['battery_entities'], dict):
    HOUSE_CONFIG['entities'].update(options['battery_entities'])
# Add similar for other merges (e.g., hot_water overrides)

# Utility functions
def parse_rates_array(rates_str):
    # Parse Octopus event state (assumes JSON-like string with rates list)
    if rates_str is None:
        logging.warning("Rates data is None—using fallback rates.")
        return []  # Will trigger fallback in get_current_rate
    try:
        rates = json.loads(rates_str) if isinstance(rates_str, str) else rates_str
        return [(r['start'], r['end'], r['value_inc_vat']) for r in rates.get('rates', [])]  # (start, end, price)
    except Exception as e:
        logging.error(f"Rate parse error: {e}")
        return []

def get_current_rate(rates):
    now = datetime.now()
    for start, end, price in rates:
        if datetime.fromisoformat(start) <= now < datetime.fromisoformat(end):
            return price / 100  # Convert pence to £
    return HOUSE_CONFIG['fallback_rates']['standard']  # Fallback

def calc_solar_gain(config, production):
    # Simple proxy: 50% of production as usable gain (adjust for your setup)
    return production * 0.5

def calc_room_loss(config, room, delta_temp, chill_factor=1.0):
    area = config['rooms'].get(room, 0)
    facing = config['facings'].get(room, 1.0)
    loss = area * max(0, delta_temp) * facing * chill_factor / 10  # Simplified U-value proxy
    return loss

def total_loss(config, ext_temp, target_temp=21.0, chill_factor=1.0):
    delta = target_temp - ext_temp
    return sum(calc_room_loss(config, room, delta, chill_factor) for room in config['rooms'])

def build_dfan_graph(config):
    G = nx.Graph()
    for room in config['rooms']:
        G.add_node(room, area=config['rooms'][room], facing=config['facings'][room])
    # Add edges for adjacent rooms if needed (e.g., heat flow); mock simple connections
    G.add_edges_from([('lounge', 'hall'), ('open_plan_ground', 'utility')])  # Expand as needed
    return G

# RL components (expanded Actor-Critic)
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

def train_rl(graph, states, config, model, optimizer, episodes=500):  # Bumped episodes
    # Mock initial training; in real, simulate episodes with rewards (cost, comfort penalties)
    for _ in range(episodes):
        # Simulate action (e.g., flow adjustment), compute reward (e.g., -cost + cop_bonus)
        action = model.actor(states)
        reward = random.uniform(-1, 1)  # Placeholder
        value = model.critic(states)
        loss = (reward - value).pow(2).mean()  # Simple MSE
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logging.info("Initial RL training complete.")

# Sim step (with improvements integrated)
def sim_step(graph, states, config, model, optimizer):
    try:
        dfan_control = fetch_ha_entity(config['entities']['dfan_control_toggle']) == 'on'
        ext_temp = float(fetch_ha_entity(config['entities']['outdoor_temp']) or 0.0)
        # Pull wind for chill_factor
        wind_speed = float(fetch_ha_entity(config['entities']['forecast_weather'], 'wind_speed') or 0.0)  # km/h
        chill_factor = 1.0
        target_temp = 21.0  # Configurable?
        delta = target_temp - ext_temp
        if wind_speed > 5:  # Formula valid above ~5 km/h
            effective_temp = 13.12 + 0.6215 * ext_temp - 11.37 * wind_speed**0.16 + 0.3965 * ext_temp * wind_speed**0.16
            chill_delta = max(0, ext_temp - effective_temp)
            chill_factor = 1.0 + (chill_delta / max(1, delta))  # Amplifies loss by wind chill ratio
        logging.info(f"Computed chill_factor: {chill_factor:.2f} based on wind {wind_speed} km/h")
        # Pull forecast for min temp and proactive
        forecast = fetch_ha_entity(config['entities']['forecast_weather'], 'forecast') or []  # List of dicts: {'datetime': ..., 'temperature': ...}
        forecast_temps = [f['temperature'] for f in forecast if 'temperature' in f and (datetime.fromisoformat(f['datetime']) - datetime.now()) < timedelta(hours=24)]
        forecast_min_temp = min(forecast_temps) if forecast_temps else ext_temp
        upcoming_cold = any(f['temperature'] < 5 for f in forecast if 'temperature' in f and (datetime.fromisoformat(f['datetime']) - datetime.now()) < timedelta(hours=12))
        operation_mode = fetch_ha_entity(config['entities']['water_heater'], 'operation_mode') or 'heat_pump'
        tank_temp = float(fetch_ha_entity(config['entities']['water_heater'], 'current_temperature') or 12.5)
        hot_water_active = 1 if operation_mode == 'high_demand' else 0
        water_load = config['hot_water']['load_kw'] if hot_water_active else 0
        hp_chosen = fetch_ha_entity(config['entities']['hp_water_tonight']) == 'on'
        current_hour = datetime.now().hour
        hp_water_night = 1 if hp_chosen and ext_temp > config['hot_water']['ext_threshold'] and config['hot_water']['cycle_start_hour'] <= current_hour < config['hot_water']['cycle_end_hour'] else 0

        # Zone offsets (pull independent sensors, compute deviation from target)
        zone_offsets = {}
        offset_loss = 0.0
        for zone, sensor_key in config['zone_sensor_map'].items():
            sensor_entity = config['entities'].get(sensor_key)
            if sensor_entity:
                zone_temp = float(fetch_ha_entity(sensor_entity) or target_temp)
                offset = target_temp - zone_temp
                zone_offsets[zone] = offset
                offset_loss += abs(offset)  # Penalty for deviations

        # Tariffs (combine current/next day; similar for exports)
        current_day_rates = fetch_ha_entity(config['entities']['current_day_rates'])
        next_day_rates = fetch_ha_entity(config['entities']['next_day_rates'])
        all_rates = parse_rates_array(current_day_rates) + parse_rates_array(next_day_rates)
        current_rate = get_current_rate(all_rates)
        # Next cheap slot: Find min rate in next 24h
        next_cheap = min(price for _, _, price in all_rates) / 100 if all_rates else config['fallback_rates']['cheap']

        # Total demand (with offsets, chill; add solar gain) — Moved earlier
        production = float(fetch_ha_entity(config['entities']['solar_production']) or 0)
        base_loss = total_loss(config, ext_temp, target_temp, chill_factor)
        total_demand = base_loss + offset_loss + water_load - calc_solar_gain(config, production)

        # Battery (compute capacity, stored, available; enhanced logic)
        soc = float(fetch_ha_entity(config['entities']['battery_soc']) or 50.0)
        design_ah = float(fetch_ha_entity(config['entities']['battery_design_capacity_ah']) or 100.0)
        remaining_ah = float(fetch_ha_entity(config['entities']['battery_remaining_capacity_ah']) or 50.0)
        capacity_kwh = design_ah * config['battery']['voltage'] / 1000
        energy_stored = remaining_ah * config['battery']['voltage'] / 1000
        discharge_available = max(0, (soc - config['battery']['min_soc_reserve']) / 100 * capacity_kwh)
        battery_power = float(fetch_ha_entity(config['entities']['battery_power']) or 0)  # Positive discharge
        charge_rate = 0.0
        discharge_rate = 0.0
        excess_solar = max(0, production - (water_load))  # Excess after basic load; refine with house consumption
        if current_rate < 0.15 and soc < 80 and excess_solar > 0:
            charge_rate = min(config['battery']['max_rate'], excess_solar / config['battery']['efficiency'])
            # Call HA service to set charge if GivEnergy supports (e.g., number.givtcp_charge_target)
            logging.info(f"Charging battery at {charge_rate:.2f} kW during cheap slot.")
        elif current_rate > 0.30 and discharge_available > 0:
            discharge_rate = min(config['battery']['max_rate'], discharge_available)
            logging.info(f"Discharging battery at {discharge_rate:.2f} kW during peak.")
        # Adjust demand/net with battery
        total_demand_adjusted = max(0, total_demand - discharge_rate) + (charge_rate / config['battery']['efficiency'])

        # Inverter/grid (net gen, import/export; check voltage)
        ac_charge = float(fetch_ha_entity(config['entities']['ac_charge_power']) or 0)
        grid_power = float(fetch_ha_entity(config['entities']['grid_power']) or 0)  # Positive import
        grid_voltage = float(fetch_ha_entity(config['entities']['grid_voltage_2']) or 230.0)
        if not (config['grid']['min_voltage'] <= grid_voltage <= config['grid']['max_voltage']):
            logging.warning(f"Grid voltage {grid_voltage}V out of bounds—pausing adjustments.")
            return
        inverter_efficiency = config['inverter']['fallback_efficiency']
        net_gen = ac_charge * inverter_efficiency  # Placeholder; expand with solar/battery flows
        net_import = max(0, grid_power)
        net_export = max(0, -grid_power)

        # CoP
        live_cop = float(fetch_ha_entity(config['entities']['hp_cop']) or 3.5)

        # Optimal flow/mode (with proactive boost)
        flow_min = float(fetch_ha_entity(config['entities']['flow_min_temp']) or 32.0)
        flow_max = float(fetch_ha_entity(config['entities']['flow_max_temp']) or 50.0)
        optimal_flow = max(flow_min, min(flow_max, 35 + (total_demand / config['peak_loss'] * (flow_max - 35))))
        optimal_mode = 'heat' if total_demand > 1.5 or ext_temp < 5 else 'off' if excess_solar > 1 or hot_water_active else 'auto'
        if upcoming_cold and current_rate < 0.15:
            optimal_flow += 5  # Pre-boost flow
            optimal_mode = 'heat'
            logging.info("Proactive heating enabled due to forecast cold snap.")

        # Update states (expanded for RL: tariff, soc, cop, flow, demand, solar, wind, forecast_min)
        states = torch.tensor([current_rate, soc, live_cop, optimal_flow, total_demand, excess_solar, wind_speed, forecast_min_temp], dtype=torch.float32)

        # Log tank suggestion
        if tank_temp < config['hot_water']['tank_low_threshold'] and current_rate < 0.15:
            logging.info("Tank low—suggest activating hot water in current cheap slot.")

        # Pause if hot water active
        if hot_water_active or hp_water_night:
            logging.info("Hot water cycle active—pausing space heating sets.")
            return

        # Apply settings if control on
        if dfan_control:
            # Set Tado for all rooms (use offsets for per-room temp if desired)
            for room in config['rooms']:
                entity_key = room + '_temp_set_hum'
                if entity_key in config['entities']:
                    entity = config['entities'][entity_key]
                    data = {'entity_id': entity, 'temperature': target_temp + zone_offsets.get(room, 0)}
                    set_ha_service('climate', 'set_temperature', data)

            # Set flow (merge base_data)
            flow_data = {'device_id': config['hp_flow_service']['device_id'],
                         **config['hp_flow_service']['base_data'],
                         'weather_comp_min_temperature': flow_min,
                         'weather_comp_max_temperature': flow_max,
                         'fixed_flow_temperature': optimal_flow}
            set_ha_service(config['hp_flow_service']['domain'], config['hp_flow_service']['service'], flow_data)

            # Set mode (use water_heater as entity for climate if appropriate; adjust if separate)
            mode_data = {'entity_id': config['entities']['water_heater'], 'hvac_mode': optimal_mode}
            set_ha_service('climate', 'set_hvac_mode', mode_data)
        else:
            logging.info(f"Shadow mode: DFAN would set flow {optimal_flow:.1f}°C and mode {optimal_mode}.")

        # RL update (enhanced reward: -rate*demand/cop + export + cop_bonus - offsets + battery arbitrage)
        action = model.actor(states.unsqueeze(0))  # e.g., [flow_delta, mode_index]
        reward = -current_rate * total_demand / live_cop + (net_export * config['fallback_rates']['export']) - (abs(charge_rate) * (1 - config['battery']['efficiency']))
        reward += (live_cop - 3.0) * 0.5 - (offset_loss * 0.1)  # CoP bonus, offset penalty
        reward += (charge_rate * (next_cheap - current_rate)) if charge_rate > 0 else - (discharge_rate * current_rate)  # Arbitrage
        value = model.critic(states.unsqueeze(0))
        loss = (reward - value).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logging.info(f"RL update: Reward {reward:.2f}, Loss {loss.item():.4f}")
    except Exception as e:
        logging.error(f"Sim step error: {e}")

# Main run
graph = build_dfan_graph(HOUSE_CONFIG)
state_dim = 8  # Expanded: tariff, soc, cop, flow, demand, solar, wind, forecast_min
action_dim = 2  # e.g., flow adjust, mode select (expand to 3 for battery if needed)
model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
states = torch.zeros(state_dim)  # Initial

train_rl(graph, states, HOUSE_CONFIG, model, optimizer)  # Initial training

def live_loop(graph, states, config, model, optimizer):
    while True:
        sim_step(graph, states, config, model, optimizer)
        time.sleep(600)  # 10 min

live_loop(graph, states, HOUSE_CONFIG, model, optimizer)