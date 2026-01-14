import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import random
from datetime import datetime, timedelta
import time
import logging
import os
import json
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# HA API URL and token from env (set in add-on options or env)
HA_URL = os.getenv('HA_URL', 'http://supervisor/core/api')  # Internal HA add-on URL
HA_TOKEN = os.getenv('HA_TOKEN', 'YOUR_LONG_LIVED_TOKEN')

def fetch_ha_entity(entity_id, attr=None):
    headers = {'Authorization': f"Bearer {HA_TOKEN}"}
    try:
        r = requests.get(f"{HA_URL}/states/{entity_id}", headers=headers)
        data = r.json()
        return data['attributes'].get(attr) if attr else data['state']
    except Exception as e:
        logging.error(f"HA pull error for {entity_id}: {e}")
        return None

def set_ha_service(domain, service, data):
    headers = {'Authorization': f"Bearer {HA_TOKEN}"}
    entity_id = data['entity_id']
    if isinstance(entity_id, list):
        for eid in entity_id:
            data_single = data.copy()
            data_single['entity_id'] = eid
            try:
                r = requests.post(f"{HA_URL}/services/{domain}/{service}", headers=headers, json=data_single)
                if r.status_code != 200: logging.error(f"HA set error: {r.text}")
            except Exception as e:
                logging.error(f"HA set error for {eid}: {e}")
    else:
        try:
            r = requests.post(f"{HA_URL}/services/{domain}/{service}", headers=headers, json=data)
            if r.status_code != 200: logging.error(f"HA set error: {r.text}")
        except Exception as e:
            logging.error(f"HA set error for {entity_id}: {e}")

# Load user options from HA add-on path
try:
    with open('/data/options.json', 'r') as f:
        user_options = json.load(f)
except Exception as e:
    logging.warning(f"Failed to load options.json: {e}. Using defaults.")
    user_options = {}

# Default config with your entities (user options override)
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
        # ... all other your Tado rooms as before
        'independent_sensor01': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor01_temperature',
        'independent_sensor02': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor02_temperature',
        'independent_sensor03': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor03_temperature',
        'independent_sensor04': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor04_temperature',
        'battery_soc': 'sensor.givtcp_ce2029g082_soc',
        'battery_design_capacity_ah': 'sensor.givtcp_dx2327m548_battery_design_capacity',
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
    'hot_water': {'load_kw': 2.5, 'ext_threshold': 3.0, 'cycle_start_hour': 0, 'cycle_end_hour': 6},
    'battery': {'min_soc_reserve': 4.0, 'efficiency': 0.9, 'voltage': 51.0, 'max_rate': 3.0},
    'grid': {'nominal_voltage': 230.0, 'min_voltage': 200.0, 'max_voltage': 250.0},
    'fallback_rates': {'cheap': 0.1495, 'standard': 0.3048, 'peak': 0.4572, 'export': 0.15},
    'inverter': {'fallback_efficiency': 0.95},
