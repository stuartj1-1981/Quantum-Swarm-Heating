import networkx as nx\
import torch\
import torch.nn as nn\
import torch.optim as optim\
import random\
from datetime import datetime, timedelta\
import time\
import logging\
\
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')\
\
# Default config with your entities (user overrides via options)\
DEFAULT_CONFIG = \{\
    'rooms': \{ 'lounge': 19.48, 'open_plan_ground': 42.14, 'utility': 3.40, 'cloaks': 2.51,\
        'bed1': 18.17, 'bed2': 13.59, 'bed3': 11.07, 'bed4': 9.79, 'bathroom': 6.02, 'ensuite1': 6.38, 'ensuite2': 3.71,\
        'hall': 9.15, 'landing': 10.09 \},\
    'facings': \{ 'lounge': 0.2, 'open_plan_ground': 1.0, 'utility': 0.5, 'cloaks': 0.5,\
        'bed1': 0.2, 'bed2': 1.0, 'bed3': 0.5, 'bed4': 0.5, 'bathroom': 0.2, 'ensuite1': 0.5, 'ensuite2': 1.0,\
        'hall': 0.2, 'landing': 0.2 \},\
    'entities': \{ \
        'lounge_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va4240580352',\
        'open_plan_ground_temp_set_hum': ['climate.tado_smart_radiator_thermostat_va0349246464', 'climate.tado_smart_radiator_thermostat_va3553629184'],\
        # ... all other your Tado rooms as before\
        'independent_sensor01': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor01_temperature',\
        'independent_sensor02': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor02_temperature',\
        'independent_sensor03': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor03_temperature',\
        'independent_sensor04': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor04_temperature',\
        'battery_soc': 'sensor.givtcp_ce2029g082_soc',\
        'battery_design_capacity_ah': 'sensor.givtcp_dx2327m548_battery_design_capacity',\
        'battery_remaining_capacity_ah': 'sensor.givtcp_dx2327m548_battery_remaining_capacity',\
        'battery_power': 'sensor.givtcp_ce2029g082_battery_power',\
        'battery_voltage': 'sensor.givtcp_ba2027g052_battery_voltage_2',\
        'ac_charge_power': 'sensor.givtcp_ce2029g082_ac_charge_power',\
        'battery_to_grid': 'sensor.givtcp_ce2029g082_battery_to_grid',\
        'battery_to_house': 'sensor.givtcp_ce2029g082_battery_to_house',\
        'grid_voltage_2': 'sensor.givtcp_ce2029g082_grid_voltage_2',\
        'grid_power': 'sensor.givtcp_ce2029g082_grid_power',\
        'current_day_rates': 'event.octopus_energy_electricity_21l3885048_2700002762631_current_day_rates',\
        'next_day_rates': 'event.octopus_energy_electricity_21l3885048_2700002762631_next_day_rates',\
        'current_day_export_rates': 'event.octopus_energy_electricity_21l3885048_2700006856140_export_current_day_rates',\
        'next_day_export_rates': 'event.octopus_energy_electricity_21l3885048_2700006856140_export_next_day_rates',\
        'solar_production': 'sensor.envoy_122019031249_current_power_production',\
        'outdoor_temp': 'sensor.front_door_temperature_measurement',\
        'forecast_weather': 'weather.home',\
        'hp_output': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_live_heat_output',\
        'hp_energy_rate': 'sensor.shellyem_c4d8d5001966_channel_1_power',\
        'total_heating_energy': 'sensor.shellyem_c4d8d5001966_channel_1_energy',\
        'hp_water_tonight': 'input_boolean.hp_chosen_for_tonight',\
        'water_heater': 'water_heater.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31',\
        'flow_min_temp': 'input_number.flow_min_temperature',\
        'flow_max_temp': 'input_number.flow_max_temperature',\
        'hp_cop': 'sensor.live_cop_calc',\
        'dfan_control_toggle': 'input_boolean.dfan_control'\
    \},\
    'zone_sensor_map': \{ 'hall': 'independent_sensor01', 'bed1': 'independent_sensor02', 'landing': 'independent_sensor03', 'open_plan_ground': 'independent_sensor04',\
        'utility': 'independent_sensor01', 'cloaks': 'independent_sensor01', 'bed2': 'independent_sensor02', 'bed3': 'independent_sensor03', 'bed4': 'independent_sensor03',\
        'bathroom': 'independent_sensor03', 'ensuite1': 'independent_sensor02', 'ensuite2': 'independent_sensor03', 'lounge': 'independent_sensor01' \},\
    'hot_water': \{'load_kw': 2.5, 'ext_threshold': 3.0, 'cycle_start_hour': 0, 'cycle_end_hour': 6\},\
    'battery': \{'min_soc_reserve': 4.0, 'efficiency': 0.9, 'voltage': 51.0, 'max_rate': 3.0\},\
    'grid': \{'nominal_voltage': 230.0, 'min_voltage': 200.0, 'max_voltage': 250.0\},\
    'fallback_rates': \{'cheap': 0.1495, 'standard': 0.3048, 'peak': 0.4572, 'export': 0.15\},\
    'inverter': \{'fallback_efficiency': 0.95\},\
    'hp_flow_service': \{\
        'domain': 'octopus_energy',\
        'service': 'set_heat_pump_flow_temp_config',\
        'device_id': 'b680894cd18521f7c706f1305b7333ea',\
        'base_data': \{\
            'weather_comp_enabled': False\
        \}\
    \},\
    'hp_hvac_service': \{\
        'domain': 'climate',\
        'service': 'set_hvac_mode',\
        'device_id': 'b680894cd18521f7c706f1305b7333ea'\
    \}\
\}\
\
# Load user options (configurable entities)\
options = hass.config_entries.async_get_options(hass.config_entries.entry_id)\
# Merge user options with defaults (e.g., user tado_rooms list overrides)\
if 'tado_rooms' in options:\
    HOUSE_CONFIG['entities'].update(\{item['room'] + '_temp_set_hum': item['entity'] for item in options['tado_rooms']\})\
# ... similar merge for independent_sensors (update 'entities' independents), battery_entities (update battery keys), etc.\
\
# ... full funcs: parse_rates_array, get_current_rate, calc_solar_gain, calc_room_loss, total_loss, build_dfan_graph, SimpleQNet, ActorCritic, train_rl as before\
\
# Sim step (with all integrations, shadow if toggle off)\
def sim_step(graph, states, config, optimizer):\
    dfan_control = fetch_ha_entity(config['entities']['dfan_control_toggle']) == 'on'\
    # Pull ext_temp from outdoor_temp\
    ext_temp = fetch_ha_entity(config['entities']['outdoor_temp']) or 0.0\
    forecast_day = 'today'\
    # Pull hot water mode/tank_temp\
    operation_mode = fetch_ha_entity(config['entities']['water_heater'], 'operation_mode') or 'heat_pump'\
    tank_temp = fetch_ha_entity(config['entities']['water_heater'], 'current_temperature') or 12.5\
    hot_water_active = 1 if operation_mode == 'high_demand' else 0\
    water_load = config['hot_water']['load_kw'] if hot_water_active else 0\
    # Pull hp_water_night boolean\
    hp_chosen = fetch_ha_entity(config['entities']['hp_water_tonight']) == 'on'\
    current_hour = datetime.now().hour\
    hp_water_night = 1 if hp_chosen and ext_temp > config['hot_water']['ext_threshold'] and config['hot_water']['cycle_start_hour'] <= current_hour < config['hot_water']['cycle_end_hour'] else 0\
    # ... pull independents for offsets, compute zone_offsets as before\
    # Pull tariff/import/export rates arrays, parse for current_rate/next_cheap as before\
    # Pull battery SOC/design_ah/remaining_ah/power/voltage, compute capacity_kwh/energy_stored/discharge_available as before\
    # Pull inverter power/efficiency/status, net_gen as before (standalone AC)\
    # Pull grid_power, net_import/export as before\
    # Pull CoP from hp_cop for r\
    live_cop = fetch_ha_entity(config['entities']['hp_cop']) or 3.5\
    # Pull solar production for excess_solar\
    production = fetch_ha_entity(config['entities']['solar_production']) or 0\
    # Compute total_demand from real_temp (with offsets) + water_load if active\
    total_demand = sum(calc_room_loss(config, k, ext_temp - 21, chill_factor) for k in config['rooms']) + water_load\
    # Battery charge/discharge/export with caps\
    # ... as before\
    # Compute optimal_flow capped min/max\
    flow_min = fetch_ha_entity(config['entities']['flow_min_temp']) or 32.0\
    flow_max = fetch_ha_entity(config['entities']['flow_max_temp']) or 50.0\
    optimal_flow = max(flow_min, min(flow_max, 35 + (total_demand / config['peak_loss'] * (flow_max - 35))))\
    # Compute optimal_mode\
    optimal_mode = 'heat' if total_demand > 1.5 or ext_temp < 5 else 'off' if excess_solar > 1 or hot_water_active else 'auto'\
    # Update states with all (tariff/battery/inverter/grid/CoP/flow/hot_water)\
    # ... as before\
    # Log suggestion if tank low and cheap\
    if tank_temp < config['hot_water']['tank_low_threshold'] and current_rate < 0.15:\
        logging.info("Tank low\'97suggest activating hot water in current cheap slot.")\
    # Pause if W-Plan active or hp_water_night\
    if hot_water_active or hp_water_night:\
        logging.info("Hot water cycle active\'97pausing space heating sets.")\
        return\
    # Set Tado/flow/mode if control on\
    if dfan_control:\
        # Set Tado per leaf (example for lounge)\
        data = \{'entity_id': config['entities']['lounge_temp_set_hum'], 'temperature': 21.0\}  # From a\
        set_ha_service('climate', 'set_temperature', data)\
        # ... for all leaves\
        # Set flow\
        flow_data = \{'weather_comp_enabled': False, 'weather_comp_min_temperature': flow_min, 'weather_comp_max_temperature': flow_max, 'fixed_flow_temperature': optimal_flow\}\
        set_ha_service('octopus_energy', 'set_heat_pump_flow_temp_config', flow_data)\
        # Set mode\
        mode_data = \{'hvac_mode': optimal_mode\}\
        set_ha_service('climate', 'set_hvac_mode', mode_data)\
    else:\
        logging.info("Shadow mode: DFAN would set flow \{optimal_flow\}\'b0C and mode \{optimal_mode\}.")\
\
    # Learn/update RL with r (tariff/cop/battery/export/offset/hot_water penalties/bonuses)\
    # ... as before\
\
# Main add-on run\
train_rl(graph, states, HOUSE_CONFIG)  # Initial\
live_loop(graph, states, HOUSE_CONFIG, optimizer)  # 10min}
