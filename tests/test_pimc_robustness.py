
import os
import sys
import time
import multiprocessing
import concurrent.futures

# Path setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from mahjong_logic import PIMCEngine, TILE_MAP
    # Check if Numba is enabled in Main Process
    import mahjong_logic
    print(f"笨・Main Process Numba Status: {getattr(mahjong_logic, 'USE_NUMBA', 'Unknown')}")
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def kill_worker(pid):
    """ 指定したPIDのプロセスを強制終了する (Windows対応) """
    print(f"裡・・Killing worker process {pid}...", flush=True)
    try:
        os.kill(pid, 9) # SIGKILL on Windows is TerminateProcess
    except Exception as e:
        print(f"Failed to kill {pid}: {e}")

def get_worker_pids():
    return [p.pid for p in multiprocessing.active_children()]

def test_pimc_robustness():
    # Setup Data
    hands_arr = [[0]*34 for _ in range(4)]
    hands_arr[3][0] = 3
    hands_arr[3][1] = 3
    hands_arr[3][2] = 3
    hands_arr[3][3] = 3
    hands_arr[3][4] = 1 # Single wait
    
    visible_counts = [0]*34
    candidates = [{"label": "1m", "idx": 0, "base_conf": 0.9}]
    doras_ids = []

    print("噫 Starting PIMC Robustness Test (Numba Check)\n", flush=True)

    # --- Phase 1: Normal Run (Warmup) ---
    print("[Phase 1] Initial Run (Warmup)", flush=True)
    # Using 2500 sims to check speed. If Numba works, this should be FAST (maybe <1s for 2500?)
    # Python was ~3.0s for 2500 sims. Numba should be significantly faster.
    res = PIMCEngine.run(hands_arr, visible_counts, candidates, doras_ids, num_worlds=2500)
    print(f"笨・Phase 1 Result: {[c.get('pimc_score') for c in res]}\n", flush=True)
    
    # --- Phase 2: Sabotage ---
    print("[Phase 2] Sabotage (Killing a worker)", flush=True)
    current_workers = get_worker_pids()
    if not current_workers:
        print("⚠️ No workers found? Executor might be lazy-init or shutdown.")
    else:
        target_pid = current_workers[0]
        kill_worker(target_pid)
        time.sleep(2) # Give it time to die and ensure the pool realizes it
    
    print("\n[Phase 3] Recovery Run", flush=True)
    try:
        res = PIMCEngine.run(hands_arr, visible_counts, candidates, doras_ids, num_worlds=2500)
        print(f"笨・Phase 3 Result: {[c.get('pimc_score') for c in res]}\n", flush=True)
    except Exception as e:
        print(f"笶・Phase 3 Failed with exception: {e}", flush=True)

    # --- Phase 4: Final Check ---
    print("[Phase 4] Post-Recovery Run", flush=True)
    try:
        res = PIMCEngine.run(hands_arr, visible_counts, candidates, doras_ids, num_worlds=2500)
        print(f"笨・Phase 4 Result: {[c.get('pimc_score') for c in res]}\n", flush=True)
        print("脂 SUCCESS: PIMC Engine recovered and continued working!", flush=True)
    except Exception as e:
        print(f"笶・Phase 4 Failed: {e}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    test_pimc_robustness()
