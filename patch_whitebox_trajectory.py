#!/usr/bin/env python3
"""
Add trajectory + signal-state sampling to trainer/tsc_trainer_whitebox.py so the
white-box FGSM run produces the same space-time data the RL attack already does.

Writes, alongside the existing fgsm_*.csv files:
    fgsm_attack_positions.csv      time, vehicle_id, is_fake, lane, position
    fgsm_attack_signal_state.csv   time, tls_id, controlled_lanes, state

Samplers are byte-for-byte the ones in trainer/tsc_trainer_adversarial_rl.py, and
are called at the same point in the loop: AFTER injection, BEFORE
reset_fake_vehicles() -- otherwise the fake vehicles are already gone.

Run once from the repo root:
    python3 patch_whitebox_trajectory.py
    python3 patch_whitebox_trajectory.py --revert
"""
import argparse
import os
import shutil
import sys

TARGET = "trainer/tsc_trainer_whitebox.py"
BACKUP = TARGET + ".bak_traj"

SAMPLERS = '''
    def _sample_signal_state(self, sim_time):
        """SUMO ground-truth signal state (ported from tsc_trainer_adversarial_rl.py)."""
        eng = getattr(self.world, 'eng', None)
        if eng is None or not hasattr(eng, 'trafficlight'):
            return []
        rows = []
        for inter in getattr(self.world, 'intersections', []):
            try:
                state = eng.trafficlight.getRedYellowGreenState(inter.id)
                controlled_lanes = eng.trafficlight.getControlledLanes(inter.id)
            except Exception:
                continue
            rows.append({
                'time': sim_time,
                'tls_id': inter.id,
                'controlled_lanes': ';'.join(controlled_lanes),
                'state': state,
            })
        return rows

    def _sample_vehicle_positions(self, sim_time):
        """Per-vehicle lane position (ported from tsc_trainer_adversarial_rl.py)."""
        eng = getattr(self.world, 'eng', None)
        if eng is None or not hasattr(eng, 'vehicle'):
            return []
        try:
            vehicle_lane, _ = self.world.get_vehicle_lane()
        except Exception:
            return []
        rows = []
        for v, lane in vehicle_lane.items():
            try:
                pos = eng.vehicle.getLanePosition(v)
            except Exception:
                continue
            rows.append({
                'time': sim_time,
                'vehicle_id': v,
                'is_fake': 'fake_' in str(v),
                'lane': lane,
                'position': pos,
            })
        return rows

    def test(self, drop_load=True):'''

ANCHOR_SAMPLERS = "    def test(self, drop_load=True):"

ANCHOR_INIT = """        pgd_sample_every = 20  # only log full step traces for every Nth decision"""
ADD_INIT = """        position_rows = []    # one row per (sampled step, vehicle): space-time trajectory data
        signal_rows = []      # one row per (sampled step, traffic light): ground-truth signal state
        self.trajectory_sample_limit = 600  # only sample the first N raw steps -- keeps file size sane
        pgd_sample_every = 20  # only log full step traces for every Nth decision"""

ANCHOR_SAMPLE = """                # victim reads its (now poisoned) observation and decides
                obs = [ag.get_ob() for ag in self.agents]"""
ADD_SAMPLE = """                # --- sample the space-time state WHILE the fakes still exist ---
                if i < self.trajectory_sample_limit:
                    position_rows.extend(self._sample_vehicle_positions(i))
                    signal_rows.extend(self._sample_signal_state(i))

                # victim reads its (now poisoned) observation and decides
                obs = [ag.get_ob() for ag in self.agents]"""

ANCHOR_WRITE = """        pgd_csv_path = self._write_csv(pgd_step_rows, 'fgsm_pgd_steps.csv')"""
ADD_WRITE = """        pos_csv_path = self._write_csv(position_rows, 'fgsm_attack_positions.csv')
        if pos_csv_path:
            self.logger.info(f"Vehicle position trace written to {pos_csv_path}")

        sig_csv_path = self._write_csv(signal_rows, 'fgsm_attack_signal_state.csv')
        if sig_csv_path:
            self.logger.info(f"Signal state trace written to {sig_csv_path}")

        pgd_csv_path = self._write_csv(pgd_step_rows, 'fgsm_pgd_steps.csv')"""

EDITS = [
    ("samplers",       ANCHOR_SAMPLERS, SAMPLERS),
    ("row buffers",    ANCHOR_INIT,     ADD_INIT),
    ("sampling call",  ANCHOR_SAMPLE,   ADD_SAMPLE),
    ("csv writes",     ANCHOR_WRITE,    ADD_WRITE),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(TARGET):
        print(f"ERROR: {TARGET} not found -- run from the repo root.", file=sys.stderr)
        return 1

    if a.revert:
        if not os.path.exists(BACKUP):
            print(f"ERROR: no backup at {BACKUP}", file=sys.stderr)
            return 1
        shutil.copy(BACKUP, TARGET)
        print(f"reverted {TARGET} from {BACKUP}")
        return 0

    src = open(TARGET).read()

    if "_sample_vehicle_positions" in src:
        print("Already patched (found _sample_vehicle_positions). Nothing to do.")
        return 0

    for name, anchor, _ in EDITS:
        n = src.count(anchor)
        if n != 1:
            print(f"ERROR: anchor for '{name}' matched {n} times, expected 1.\n"
                  f"       The file differs from what this patch expects; edit by hand.",
                  file=sys.stderr)
            return 1

    shutil.copy(TARGET, BACKUP)
    for name, anchor, replacement in EDITS:
        src = src.replace(anchor, replacement, 1)
        print(f"  patched: {name}")
    open(TARGET, "w").write(src)

    import py_compile
    try:
        py_compile.compile(TARGET, doraise=True)
    except py_compile.PyCompileError as e:
        shutil.copy(BACKUP, TARGET)
        print(f"ERROR: patched file does not compile, reverted.\n{e}", file=sys.stderr)
        return 1

    print(f"\nOK. Backup at {BACKUP}.")
    print("Re-run the FGSM eval; it will now also write:")
    print("  fgsm_attack_positions.csv")
    print("  fgsm_attack_signal_state.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
