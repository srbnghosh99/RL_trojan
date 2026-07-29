#!/usr/bin/env python3
"""
make_flow_variants.py
================================================================================
Create LOW and HIGH demand variants of a SUMO route file and matching sim_sumo
.cfg files, so the sweep can compare the SAME network under different traffic
levels (the "diff traffic flow: normal / high / less" axis).

The base route files are explicit vehicle lists, e.g.:
    <vehicle id="0" depart="7"><route edges="road_2_1_2 road_1_1_2"/></vehicle>

  * LOW  : keep a fraction of the vehicles (thin the demand), seeded subsample.
  * HIGH : duplicate a fraction of vehicles with new ids + jittered depart times.
  * NORMAL is just the untouched base file / existing .cfg.

Usage
-----
    python analysis/make_flow_variants.py                     # 1x1, low=0.5 high=1.6
    python analysis/make_flow_variants.py --net cityflow1x1 --low 0.5 --high 1.6
    python analysis/make_flow_variants.py --net cityflow4x4 --low 0.6 --high 1.5

Writes (for --net cityflow1x1):
    sumo_config/data/cityflow1x1/roadnet_low.rou.xml
    sumo_config/data/cityflow1x1/roadnet_high.rou.xml
    configs/sim_sumo/cityflow1x1_low.cfg
    configs/sim_sumo/cityflow1x1_high.cfg

Then run the sweep with networks: cityflow1x1_low, cityflow1x1, cityflow1x1_high
(the "network" field / signal_config key is copied from the base cfg, so the
victim agents still resolve phase_pairs correctly).
"""
import argparse
import copy
import json
import os
import random
import xml.etree.ElementTree as ET

# REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPO = "/Users/shrabanighosh/UNCC/Spring2026/RL_TSC_Backdoor-trojdrlshra"


def _load_vehicles(rou_path):
    tree = ET.parse(rou_path)
    root = tree.getroot()
    vtypes = [el for el in root if el.tag == 'vType']
    vehicles = [el for el in root if el.tag == 'vehicle']
    others = [el for el in root if el.tag not in ('vType', 'vehicle')]
    return tree, root, vtypes, vehicles, others


def _depart(el):
    try:
        return float(el.get('depart', '0'))
    except ValueError:
        return 0.0


def _write(rou_out, vtypes, vehicles, others):
    root = ET.Element('routes')
    root.set('xmlns:xsi', "http://www.w3.org/2001/XMLSchema-instance")
    root.set('xsi:noNamespaceSchemaLocation', "http://sumo.dlr.de/xsd/routes_file.xsd")
    for el in vtypes:
        root.append(copy.deepcopy(el))
    for el in others:
        root.append(copy.deepcopy(el))
    # SUMO requires vehicles sorted by increasing depart time
    for el in sorted(vehicles, key=_depart):
        root.append(el)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(rou_out, encoding='utf-8', xml_declaration=True)


def make_low(vehicles, keep, seed):
    rng = random.Random(seed)
    kept = [v for v in vehicles if rng.random() < keep]
    return [copy.deepcopy(v) for v in kept]


def make_high(vehicles, factor, seed):
    rng = random.Random(seed)
    out = [copy.deepcopy(v) for v in vehicles]
    extra = int(round(len(vehicles) * (factor - 1.0)))
    max_depart = max((_depart(v) for v in vehicles), default=3600.0)
    for k in range(extra):
        src = rng.choice(vehicles)
        dup = copy.deepcopy(src)
        dup.set('id', f'extra_{k}')
        # jitter depart a little so duplicates don't stack exactly
        dup.set('depart', f'{min(max_depart, _depart(src) + rng.uniform(0, 5)):.2f}')
        out.append(dup)
    return out


def make_cfg(base_cfg_path, out_cfg_path, new_flow_rel):
    with open(base_cfg_path) as f:
        cfg = json.load(f)
    cfg['flowFile'] = new_flow_rel
    with open(out_cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', default='cityflow1x1',
                    help='network dir under sumo_config/data and cfg name in configs/sim_sumo')
    ap.add_argument('--rou', default='roadnet.rou.xml', help='base route file name')
    ap.add_argument('--low', type=float, default=0.5, help='fraction of vehicles to keep for LOW')
    ap.add_argument('--high', type=float, default=1.6, help='demand multiplier for HIGH')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    data_dir = os.path.join(REPO, 'sumo_config', 'data', a.net)
    base_rou = os.path.join(data_dir, a.rou)
    base_cfg = os.path.join(REPO, 'configs', 'sim_sumo', f'{a.net}.cfg')
    if not os.path.exists(base_rou):
        raise SystemExit(f"route file not found: {base_rou}")
    if not os.path.exists(base_cfg):
        raise SystemExit(f"base cfg not found: {base_cfg}")

    _, _, vtypes, vehicles, others = _load_vehicles(base_rou)
    print(f"[{a.net}] base vehicles: {len(vehicles)}")

    # LOW
    low_veh = make_low(vehicles, a.low, a.seed)
    low_rou = os.path.join(data_dir, 'roadnet_low.rou.xml')
    _write(low_rou, vtypes, low_veh, others)
    make_cfg(base_cfg, os.path.join(REPO, 'configs', 'sim_sumo', f'{a.net}_low.cfg'),
             f'{a.net}/roadnet_low.rou.xml')
    print(f"  LOW  -> {len(low_veh):5d} vehicles  ({low_rou})")

    # HIGH
    high_veh = make_high(vehicles, a.high, a.seed)
    high_rou = os.path.join(data_dir, 'roadnet_high.rou.xml')
    _write(high_rou, vtypes, high_veh, others)
    make_cfg(base_cfg, os.path.join(REPO, 'configs', 'sim_sumo', f'{a.net}_high.cfg'),
             f'{a.net}/roadnet_high.rou.xml')
    print(f"  HIGH -> {len(high_veh):5d} vehicles  ({high_rou})")

    print("cfgs written:",
          f"configs/sim_sumo/{a.net}_low.cfg,", f"configs/sim_sumo/{a.net}_high.cfg")


if __name__ == '__main__':
    main()
