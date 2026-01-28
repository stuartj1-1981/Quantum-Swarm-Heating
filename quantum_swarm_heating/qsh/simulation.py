from .utils import safe_float, total_loss, calc_room_loss, shutdown_handler, get_current_temp
from .ha_integration import fetch_ha_entity, set_ha_service
from .config import HOUSE_CONFIG, ZONE_OFFSETS
from .rl_model import ActorCritic
import time
import logging
import json
import signal
import sys
from collections import deque
import numpy as np


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

# New globals for per-room nudges/drops (hysteresis, cooldown)
room_nudge_hyst = {room: 0 for room in HOUSE_CONFIG['rooms']}
room_nudge_cooldown = {room: 0 for room in HOUSE_CONFIG['rooms']}
room_nudge_accum = {room: 0.0 for room in HOUSE_CONFIG['rooms']}

# Global for persisted rates
prev_all_rates = []

# Globals for refinements
demand_history = deque(maxlen=5)
prod_history = deque(maxlen=5)
grid_history = deque(maxlen=5)
cop_history = deque([4.0] * 5, maxlen=5)
heat_up_history = deque(maxlen=5)
low_delta_persist = 0
low_power_start_time = None
prev_hp_power = 1.0
prev_flow_temp = 35.0
prev_cop = 3.5
cycle_type = None
cycle_start = None
pause_end = None
action_counter = 0
prev_flow = 35.0
prev_mode = 'off'
prev_demand = 3.5
prev_time = time.time()
prev_actual_loss = 0.0
reward_history = deque(maxlen=1000)
loss_history = deque(maxlen=1000)
pause_count = 0
undetected_count = 0
enable_plots = False  # hardcoded during Phase 1B debug; options.json removal planned in Phase 1C
first_loop = True
epsilon = 0.2
blend_factor = 0.0
last_heat_time = time.time() - 600
consecutive_slow = 0

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

def sim_step(graph, states, config, model, optimizer, action_counter, prev_flow, prev_mode, prev_demand):
    global low_delta_persist, low_power_start_time, prev_hp_power, prev_flow_temp, prev_cop, cycle_type
    global demand_history, prod_history, grid_history, prev_time, cycle_start, prev_actual_loss
    global pause_count, undetected_count, first_loop, pause_end, prev_all_rates, epsilon, blend_factor
    global cop_history, heat_up_history, last_heat_time, consecutive_slow, room_nudge_hyst, room_nudge_cooldown, room_nudge_accum

    current_time = time.time()
    time_delta = current_time - prev_time
    prev_time = current_time

    dfan_control = fetch_ha_entity(config['entities']['dfan_control_toggle'], default='off') == 'on'
    target_temp_raw = fetch_ha_entity(config['entities']['pid_target_temperature'], default=21.0)
    target_temp = safe_float(target_temp_raw, 21.0)
    logging.info(f"Using target_temp: {target_temp}°C from pid_target_temperature.")

    hot_water_state = fetch_ha_entity(config['entities']['water_heater'], default='off')
    hot_water_active = hot_water_state == 'high_demand'

    room_targets = {room: target_temp + ZONE_OFFSETS.get(room, 0.0) for room in config['rooms']}
    for room in config['persistent_zones']:
        if room in room_targets:
            room_targets[room] = 25.0

    ext_temp_raw = fetch_ha_entity(config['entities']['outdoor_temp'], default=0.0)
    ext_temp = safe_float(ext_temp_raw, 0.0)

    current_day_rates_raw = fetch_ha_entity(config['entities']['current_day_rates'], 'rates', default=[])
    current_day_rates = parse_rates_array(current_day_rates_raw, suppress_warning=(datetime.now(timezone.utc).hour < 16))

    rates_retry = 0
    while not current_day_rates and rates_retry < 3:
        rates_retry += 1
        logging.warning(f"Rates retry {rates_retry}/3...")
        current_day_rates_raw = fetch_ha_entity(config['entities']['current_day_rates'], 'rates', default=[])
        current_day_rates = parse_rates_array(current_day_rates_raw)

    current_rate = get_current_rate(current_day_rates)

    if hot_water_active:
        logging.info("Hot water active: Pausing all QSH processing (space heating optimizations, RL updates, cycle detection, and HA sets).")
        optimal_mode = 'off'
        optimal_flow = prev_flow
        total_demand_adjusted = 0.0

        logging.info(f"Mode decision: optimal_mode='off', total_demand=0.00 kW, ext_temp={ext_temp:.1f}°C, upcoming_cold=False, upcoming_high_wind=False, current_rate={current_rate:.3f} GBP/kWh, hot_water_active=True")

        low_delta_persist = 0
        low_power_start_time = None
        cycle_type = None
        cycle_start = None
        pause_end = None

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
        hp_power = safe_float(fetch_ha_entity(config['entities']['hp_energy_rate'], default=0.0), 0.0)
        upcoming_cold = False
        upcoming_high_wind = False
        actual_loss = 0.0
        sum_af = 0.0
        flow_min = safe_float(fetch_ha_entity(config['entities']['flow_min_temp'], default=32.0), 32.0)
        flow_max = safe_float(fetch_ha_entity(config['entities']['flow_max_temp'], default=50.0), 50.0)
        current_flow_temp = safe_float(fetch_ha_entity(config['entities']['hp_flow_temp'], default=35.0), 35.0)
        reward = 0.0
        loss = torch.tensor(0.0)
        heat_up_power = 0.0
        aggregate_heat_up = 0.0
        export_kw = 0.0

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

        structured_log = {
            "event": "loop",
            "reason": "hot_water_active",
            "target_temp": target_temp,
            "ext_temp": ext_temp,
            "current_rate": current_rate,
            "optimal_mode": optimal_mode,
            "optimal_flow": optimal_flow,
            "total_demand_adjusted": total_demand_adjusted,
            "dfan_control": dfan_control,
            "time_delta": time_delta,
        }
        logging.info(f"QSH_LOOP: {json.dumps(structured_log)}")

        return optimal_mode, optimal_flow, total_demand_adjusted, prev_flow, prev_mode, prev_demand

    sensor_temps = {
        'independent_sensor01': safe_float(fetch_ha_entity(config['entities'].get('independent_sensor01'), default=target_temp), target_temp),
        'independent_sensor02': safe_float(fetch_ha_entity(config['entities'].get('independent_sensor02'), default=target_temp), target_temp),
        'independent_sensor03': safe_float(fetch_ha_entity(config['entities'].get('independent_sensor03'), default=target_temp), target_temp),
        'independent_sensor04': safe_float(fetch_ha_entity(config['entities'].get('independent_sensor04'), default=target_temp), target_temp),
    }

    heating_percs = {}
    avg_open_frac = 0.0
    num_rooms = len(config['rooms'])
    if num_rooms == 0:
        logging.warning("No rooms defined in HOUSE_CONFIG—skipping heating perc fetch.")
    else:
        for room in config['rooms']:
            heating_entity = config['entities'].get(room + '_heating')
            if heating_entity:
                perc = safe_float(fetch_ha_entity(heating_entity, default=0.0), 0.0)
                heating_percs[room] = perc
                avg_open_frac += perc / 100.0
            else:
                logging.warning(f"No heating entity for {room}.")
        avg_open_frac /= num_rooms

    sum_af = sum(config['rooms'].get(room, 0) * config['facings'].get(room, 1.0) for room in config['rooms'])

    actual_loss = total_loss(config, ext_temp, room_targets, chill_factor=1.0, loss_coeff=config['peak_loss'], sum_af=sum_af)

    solar_production = safe_float(fetch_ha_entity(config['entities'].get('solar_production'), default=0.0), 0.0)
    hp_power = safe_float(fetch_ha_entity(config['entities'].get('hp_energy_rate'), default=0.0), 0.0)
    delta_t = safe_float(fetch_ha_entity(config['entities'].get('primary_diff'), default=3.0), 3.0)
    grid_power = safe_float(fetch_ha_entity(config['entities'].get('grid_power'), default=0.0), 0.0)
    soc = safe_float(fetch_ha_entity(config['entities'].get('battery_soc'), default=50.0), 50.0)
    wind_speed = 0.0
    forecast_min_temp = 0.0
    excess_solar = max(0, solar_production - actual_loss)
    upcoming_cold = False
    upcoming_high_wind = False
    export_kw = max(0, -grid_power / 1000.0) if grid_power < 0 else 0.0

    demand_history.append(actual_loss)
    prod_history.append(solar_production)
    grid_history.append(grid_power)
    smoothed_demand = np.mean(demand_history) if demand_history else 0.0
    demand_std = np.std(demand_history) if demand_history else 0.0
    smoothed_grid = np.mean(grid_history) if grid_history else 0.0

    thermal_mass_per_m2 = config['thermal_mass_per_m2']
    heat_up_tau_h = config['heat_up_tau_h']
    heat_up_power = 0.0
    aggregate_heat_up = 0.0
    for room, area in config['rooms'].items():
        current_temp = get_current_temp(room, sensor_temps, room_targets.get(room, target_temp))
        max_delta = max(0, room_targets.get(room, target_temp) - current_temp)
        heat_up_power += (area * thermal_mass_per_m2 * max_delta) / heat_up_tau_h
        aggregate_heat_up += max_delta

    logging.info(f"Calculated heat_up_power: {heat_up_power:.2f} kW")

    total_demand_adjusted = actual_loss + heat_up_power - solar_production

    reward = 0.0
    if total_demand_adjusted > 0 and delta_t < 3.0:
        reward -= 0.2 * (heat_up_power / total_demand_adjusted)

    dissipation_fired = False
    total_disp = 0.0
    if low_delta_persist >= 1 or avg_open_frac < 0.6:
        dissipation_fired = True
        low_frac_rooms = [r for r in config['rooms'] if heating_percs.get(r, 0)/100.0 < 0.6]
        energy_disp = {}
        for room in low_frac_rooms:
            current_temp = get_current_temp(room, sensor_temps, room_targets.get(room, target_temp))
            delta_temp_room = max(0, room_targets.get(room, target_temp) - current_temp)
            energy_disp[room] = config['emitter_kw'].get(room, 1.0) * delta_temp_room
            total_disp += energy_disp[room]
        prioritized_rooms = sorted(low_frac_rooms, key=lambda r: energy_disp.get(r, 0), reverse=True)
        top_opens = prioritized_rooms[:random.randint(3, 5)]
        successful_opens = 0
        nudge_shares = {r: energy_disp[r] / total_disp if total_disp > 0 else 0 for r in top_opens}
        for room in top_opens:
            nudge = nudge_shares[room] * config['nudge_budget']
            mode = config['room_control_mode'].get(room, 'indirect')
            valve_entity = f'number.qsh_{room}_valve_target'
            if mode == 'direct' and fetch_ha_entity(valve_entity, default=None) is None:
                logging.warning(f"TRV {room} missing—fallback indirect")
                mode = 'indirect'
            effective_mode = 'indirect' if (mode == 'direct' and fetch_ha_entity(valve_entity, default=None) is None) else mode

            current_temp = get_current_temp(room, sensor_temps, room_targets.get(room, target_temp))
            delta_temp_room = max(0, room_targets.get(room, target_temp) - current_temp)

            if effective_mode == 'direct':
                set_frac = min(90, 75 + (nudge * 10))
                if dfan_control:
                    set_ha_service('number', 'set_value', {'entity_id': valve_entity, 'value': set_frac})
                    successful_opens += 1
                logging.info(f"Direct kW-disp: {room} to {set_frac:.0f}% (disp={config['emitter_kw'].get(room, 1.0)*delta_temp_room:.2f}, rating={config['emitter_kw'].get(room, 1.0)})")
            else:
                new_accum = room_nudge_accum.get(room, 0.0) + nudge
                if abs(new_accum) <= config['nudge_budget']:
                    room_nudge_accum[room] = new_accum
                    room_targets[room] += nudge
                    room_nudge_cooldown[room] = 0
                    successful_opens += 1
                    logging.info(f"Indirect kW-disp: {room} +{nudge:.1f}°C (accum {new_accum:.1f}°C, disp={config['emitter_kw'].get(room, 1.0)*delta_temp_room:.2f}, rating={config['emitter_kw'].get(room, 1.0)})")

        logging.info(f"Successful opens: {successful_opens}/{len(top_opens)} (prioritized large emitters)")
        low_delta_persist = 0

    flow_min = safe_float(fetch_ha_entity(config['entities'].get('flow_min_temp'), default=32.0), 32.0)
    flow_max = safe_float(fetch_ha_entity(config['entities'].get('flow_max_temp'), default=50.0), 50.0)

    det_flow = 30 + (smoothed_demand / config['peak_loss'] * 10)
    if len(heat_up_history) > 1:
        prev_time_h, prev_heat_up = heat_up_history[0]
        mean_rate = (aggregate_heat_up - prev_heat_up) / ((current_time - prev_time_h) / 60) if (current_time - prev_time_h) > 0 else 0.0
    else:
        mean_rate = 0.0
    heat_up_history.append((current_time, aggregate_heat_up))
    if mean_rate < 0.1:
        consecutive_slow += 1
    else:
        consecutive_slow = 0
    if consecutive_slow >= 3 and avg_open_frac > 0.5:
        det_flow += 2
        logging.info(f"Slow heat_up rate ({mean_rate:.2f}°C/min)—incrementing flow +2°C.")
    if upcoming_cold and current_rate < 0.15:
        det_flow += 3
    if upcoming_high_wind and current_rate < 0.15:
        det_flow += 2
    det_flow = max(flow_min, min(flow_max, det_flow))

    det_mode = 'heat' if smoothed_demand > 0 or aggregate_heat_up > 0.2 else 'off'

    states = torch.tensor([
        current_rate, soc, prev_cop, prev_flow, smoothed_demand, excess_solar,
        wind_speed, forecast_min_temp, smoothed_grid, delta_t, hp_power, 1.0,
        demand_std, delta_t, avg_open_frac
    ], dtype=torch.float32)
    action, value = model(states.unsqueeze(0))
    action = action.squeeze(0)

    actor_flow = 30 + (torch.sigmoid(action[1]) * 20)
    optimal_flow = (blend_factor * actor_flow.item()) + ((1 - blend_factor) * det_flow)

    flow_adjust = 0.0
    upward_nudge_count = 0
    for room in config['rooms']:
        mode = config['room_control_mode'].get(room, 'indirect')
        frac = heating_percs.get(room, 0) / 100.0
        logging.info(f"Hybrid: {room} mode={mode}, frac={frac:.2f}")

        valve_entity = f'number.qsh_{room}_valve_target'
        if mode == 'direct' and fetch_ha_entity(valve_entity, default=None) is None:
            logging.warning(f"No valve entity for {room}—fallback to indirect.")
            mode = 'indirect'

        if mode == 'direct':
            target_frac = 0.75
            if frac < 0.60:
                target_frac += 0.20
            if frac > 0.85 and mean_rate < 0.1:
                target_frac -= 0.15
            target_frac = max(0.5, min(0.9, target_frac))
            if dfan_control:
                set_ha_service('number', 'set_value', {'entity_id': valve_entity, 'value': target_frac * 100})
            else:
                logging.info(f"Shadow direct: Would set {room} valve to {target_frac * 100:.0f}%")
            if frac < 0.7:
                flow_adjust -= 3.0 * (0.7 - frac)
        else:
            nudge = 0.0
            if room_nudge_cooldown.get(room, 0) > 0:
                room_nudge_cooldown[room] -= 1
                continue
            if frac < 0.4:
                room_nudge_hyst[room] += 1
                if room_nudge_hyst[room] >= 2:
                    nudge = 0.6
                    upward_nudge_count += 1
            elif frac < 0.6:
                room_nudge_hyst[room] += 1
                if room_nudge_hyst[room] >= 2:
                    nudge = 0.3
                    upward_nudge_count += 1
            elif frac > 0.95:
                room_nudge_hyst[room] += 1
                if room_nudge_hyst[room] >= 2:
                    nudge = -0.5
            elif frac > 0.85:
                room_nudge_hyst[room] += 1
                if room_nudge_hyst[room] >= 2:
                    nudge = -0.25
            else:
                room_nudge_hyst[room] = 0

            if nudge != 0.0:
                new_accum = room_nudge_accum.get(room, 0.0) + nudge
                if abs(new_accum) > config['nudge_budget']:
                    nudge = 0.0
                    logging.warning(f"{room} nudge clamped at ±{config['nudge_budget']}°C.")
                else:
                    room_nudge_accum[room] = new_accum
                    room_targets[room] += nudge
                    room_nudge_cooldown[room] = 5
                    room_nudge_hyst[room] = 0
                    logging.info(f"Indirect nudge: {room} +{nudge:.2f}°C (accum {room_nudge_accum[room]:.2f}°C)")
            if frac < 0.7:
                flow_adjust -= 5.0 * (0.7 - frac)

    if upward_nudge_count > 0:
        has_direct = any(config['room_control_mode'].get(r, 'indirect') == 'direct' for r in config['rooms'])
        drop_base = -3.0 if has_direct else -5.0
        flow_adjust += drop_base + (1.0 if has_direct else 1.5) * upward_nudge_count
        flow_adjust = max(-6.0, flow_adjust)
        logging.info(f"Hybrid dissipation: Flow adjust {flow_adjust:.1f}°C (upward nudges={upward_nudge_count}, base={drop_base})")
    optimal_flow += flow_adjust
    has_direct_any = any(config['room_control_mode'].get(r, 'indirect') == 'direct' for r in config['rooms'])
    optimal_flow = max(30.0 if has_direct_any else 28.0, min(flow_max, optimal_flow))

    optimal_flow = max(prev_flow - 3, min(prev_flow + 3, optimal_flow))

    optimal_mode = det_mode
    if optimal_mode == 'heat':
        last_heat_time = current_time
    elif optimal_mode == 'off' and (current_time - last_heat_time) < 600:
        optimal_mode = 'heat'
        logging.info("Min run safeguard: Extending 'heat' to dissipate residual.")

    if soc > 80 and export_kw > 1 and current_rate > 0.3 and aggregate_heat_up <= 0.2:
        optimal_mode = 'off'
        logging.info("Export optimized pause: high SOC and export during peak rate—overriding to 'off'")

    optimal_flow = round(optimal_flow, 1)
    flow_min = round(flow_min, 1)
    flow_max = round(flow_max, 1)
    logging.info(f"Rounded optimal_flow: {optimal_flow:.1f}°C, flow_min: {flow_min:.1f}°C, flow_max: {flow_max:.1f}°C")

    flow_delta = abs(optimal_flow - prev_flow)
    flow_ramp_rate = abs(flow_delta) / (time_delta / 60) if time_delta > 0 else 0

    current_flow_temp = safe_float(fetch_ha_entity(config['entities'].get('hp_flow_temp'), default=35.0), 35.0)
    cop_value = fetch_ha_entity(config['entities'].get('hp_cop'), default=None)
    if cop_value == 'unavailable':
        cop_value = fetch_ha_entity(config['entities'].get('hp_cop'), default=None)
    if cop_value in ('unavailable', None) or safe_float(cop_value, 0.0) <= 0:
        logging.warning("COP gap or <=0 - using median history.")
        non_zero_cops = [c for c in cop_history if c > 0]
        live_cop = np.median(non_zero_cops) if non_zero_cops else 3.0
    else:
        live_cop = safe_float(cop_value, 3.0)
    cop_history.append(live_cop)

    power_delta = hp_power - prev_hp_power
    flow_delta_actual = current_flow_temp - prev_flow_temp
    cop_delta = live_cop - prev_cop
    loss_delta = actual_loss - prev_actual_loss
    demand_delta = total_demand_adjusted - prev_demand

    if pause_end and current_time < pause_end:
        time_remaining = pause_end - current_time
        logging.info(f"Ongoing cycle pause: Type {cycle_type}, remaining {time_remaining:.0f}s—pausing adjustments.")
        pause_count += 1
        prev_hp_power = hp_power
        prev_flow_temp = current_flow_temp
        prev_cop = live_cop
        prev_actual_loss = actual_loss
        prev_demand = total_demand_adjusted
        structured_log = {
            "event": "loop",
            "reason": "cycle_pause",
            "target_temp": target_temp,
            "ext_temp": ext_temp,
            "current_rate": current_rate,
            "optimal_mode": prev_mode,
            "optimal_flow": prev_flow,
            "total_demand_adjusted": prev_demand,
            "dfan_control": dfan_control,
            "time_delta": time_delta,
        }
        logging.info(f"QSH_LOOP: {json.dumps(structured_log)}")
        return prev_mode, prev_flow, prev_demand, prev_flow, prev_mode, prev_demand

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
            pause_end = current_time + EXTENDED_RECOVERY_TIME

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

    prev_hp_power = hp_power
    prev_flow_temp = current_flow_temp
    prev_cop = live_cop
    prev_actual_loss = actual_loss
    prev_demand = total_demand_adjusted

    logging.info(
        f"Mode decision: optimal_mode='{optimal_mode}', total_demand={smoothed_demand:.2f} kW, "
        f"ext_temp={ext_temp:.1f}°C, upcoming_cold={upcoming_cold}, upcoming_high_wind={upcoming_high_wind}, "
        f"current_rate={current_rate:.3f} GBP/kWh, hot_water_active={hot_water_active}"
    )

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

    if abs(demand_delta) < 0.5:
        reward += 0.3
    if abs(flow_delta) < 0.5:
        reward += 0.2

    volatility_penalty = 0.0
    if demand_std > 0.5:
        volatility_penalty = 0.2 * (demand_std - 0.5)
    reward -= volatility_penalty
    logging.info(f"DFAN RL: demand_std={demand_std:.2f} kW, volatility_penalty={volatility_penalty:.2f}")

    if ext_temp > 5 and optimal_flow > 40:
        reward -= 1.5
    if avg_open_frac < 0.5:
        penalty = 2.5 if has_direct_any else 2.0
        reward -= penalty

    original_cop = safe_float(fetch_ha_entity(config['entities'].get('hp_cop'), default=0.0), 0.0)
    if original_cop <= 0:
        logging.info("Skipping reward update due to COP <=0.")
        reward = 0.0
    td_error = reward - value.item()
    critic_loss = td_error ** 2

    norm_flow = (optimal_flow - 30) / 20
    flow_mse = (action[1] - torch.tensor(norm_flow)).pow(2)
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
    loss_history.append(loss.item())

    epsilon = max(0.05, epsilon * 0.995)

    if len(reward_history) >= 10 and np.mean(list(reward_history)[-10:]) > 0:
        blend_factor = min(1.0, blend_factor + 0.01)

    flow_delta = abs(optimal_flow - prev_flow)
    demand_delta_abs = abs(smoothed_demand - prev_demand)
    urgent = (optimal_mode != prev_mode) or (flow_delta > 2.0) or (demand_delta_abs > 0.5) or (low_delta_persist >= 2) or (hp_power < 0.20)
    if dfan_control and (action_counter % 10 == 0 or urgent):
        for room in config['rooms']:
            entity_key = room + '_temp_set_hum'
            valve_entity = f'number.qsh_{room}_valve_target'
            config_mode = config['room_control_mode'].get(room, 'indirect')
            if entity_key in config['entities'] and (config_mode == 'indirect' or (config_mode == 'direct' and fetch_ha_entity(valve_entity, default=None) is None)):
                temperature = room_targets[room]
                data = {'entity_id': config['entities'][entity_key], 'temperature': temperature}
                set_ha_service('climate', 'set_temperature', data)

        flow_data = {
            'device_id': config['hp_flow_service']['device_id'],
            **config['hp_flow_service']['base_data'],
            'weather_comp_min_temperature': flow_min,
            'weather_comp_max_temperature': flow_max,
            'fixed_flow_temperature': optimal_flow
        }
        set_ha_service(config['hp_flow_service']['domain'], config['hp_flow_service']['service'], flow_data)

        mode_data = {'device_id': config['hp_hvac_service']['device_id'], 'hvac_mode': optimal_mode}
        set_ha_service(config['hp_hvac_service']['domain'], config['hp_hvac_service']['service'], mode_data)

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

    structured_log = {
        "event": "loop",
        "reason": "normal",
        "target_temp": target_temp,
        "ext_temp": ext_temp,
        "current_rate": current_rate,
        "optimal_mode": optimal_mode,
        "optimal_flow": optimal_flow,
        "total_demand_adjusted": total_demand_adjusted,
        "dfan_control": dfan_control,
        "time_delta": time_delta,
    }
    logging.info(f"QSH_LOOP: {json.dumps(structured_log)}")

    return optimal_mode, optimal_flow, total_demand_adjusted, prev_flow, prev_mode, prev_demand