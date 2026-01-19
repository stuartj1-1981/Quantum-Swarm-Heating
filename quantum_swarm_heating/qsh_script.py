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
from collections import defaultdict  # Added for zone grouping

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
    headers = {'Authorization': f"Bearer {HA_TOKEN}"}
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

# Hardcoded HOUSE_CONFIG for your specific setup (full Tado entities added; peak_loss=5.0 from -3°C calc)
# Updated: Renamed 'open_plan_ground' to 'open_plan' for entity name consistency (avoids _ground in shadow setpoints)
# Added design_target and peak_ext for calibrated loss_coeff
# Added thermal_mass_per_m2 and heat_up_tau_h for thermal mass modeling
# Reverted: peak_loss to 5.0 kW @ -3°C based on real world data from previous conversations
# Removed: hot_water config and unnecessary battery/hot water entities (status-only fetches retained where needed)
# Re-added: 'water_heater' entity for status-only detection of 'high_demand' mode
# Added: 'grid_power' entity as per latest integration
# New: 'persistent_zones' for always-open balancing (bathroom/ensuites)
# New: 'primary_diff' for ΔT safeguards
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
        'landing_temp_set_hum': 'climate