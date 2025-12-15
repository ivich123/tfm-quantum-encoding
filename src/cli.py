import argparse, yaml, os, sys, datetime
from src.train import run_experiment

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", action='append', default=[])  # action='append' for multiple --override
    args = ap.parse_args()

    cfg = {}
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # overrides "a.b=c" → cfg["a"]["b"] = cast(c)
    for s in args.override:
        k, v = s.split("=", 1)
        path = k.split(".")
        cur = cfg
        for p in path[:-1]:
            cur = cur.setdefault(p, {})
        # improved basic cast
        if v.lower() in ("true", "false"):
            v = v.lower() == "true"
        elif v.isdigit():
            v = int(v)
        else:
            try:
                v = float(v)
            except:
                pass  # keep as string
        cur[path[-1]] = v

    # Setup Logging
    log_dir = os.path.join(cfg.get("log_dir", "outputs"), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    exp_name = cfg.get("experiment", "exp")
    model_type = cfg.get("model", {}).get("type", "unknown")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{exp_name}_{model_type}_{timestamp}.log")
    
    print(f"Writing log to: {log_file}")
    sys.stdout = Tee(log_file, "a")

    try:
        run_experiment(cfg)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        # raise # Optional: re-raise if you want full traceback in console/log

if __name__ == "__main__":
    main()
