import torch
import pennylane as qml
import sys
import platform
import os

def check_system():
    print("="*60)
    print("🔍 SYSTEM DIAGNOSTICS")
    print("="*60)
    
    # 1. System Info
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    
    # 2. PyTorch
    print("\n[PyTorch]")
    print(f"Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        n_gpus = torch.cuda.device_count()
        print(f"GPU Count: {n_gpus}")
        for i in range(n_gpus):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            
        # Functional Test
        try:
            print("  - Functional Check: ", end="")
            # Try to allocate a tensor on GPU and perform computation
            dummy = torch.zeros(1).cuda()
            _ = dummy * 2
            del dummy
            print("✓ OK (Tensor allocation & execution successful)")
            
            # Memory check
            free, total = torch.cuda.mem_get_info()
            print(f"  - Memory: {free/1024**3:.2f} GB free / {total/1024**3:.2f} GB total")
            
        except Exception as e:
            print("❌ FAILED")
            print(f"    ⚠ Error: {e}")
            print("    ⚠ NOTE: CUDA is installed but not working (likely version mismatch). PyTorch will fallback to CPU.")
    else:
        print("CUDA Available: No (running on CPU)")

    # 3. PennyLane
    print("\n[PennyLane]")
    print(f"Version: {qml.__version__}")
    
    # Check key plugins
    print("Key Plugins:")
    plugins = qml.plugin_devices
    interesting_plugins = ["lightning.qubit", "lightning.gpu", "lightning.kokkos", "default.qubit"]
    
    for p in interesting_plugins:
        status = "INSTALLED" if p in plugins else "NOT FOUND"
        print(f"  - {p:<20} : {status}")

    # 4. Device Functional Test
    print("\n[Device Functional Test & Benchmark]")
    print(f"{'Device':<20} | {'Status':<10} | {'Time (ms)':<10}")
    print("-" * 45)
    
    devices_to_test = ["default.qubit"]
    if "lightning.qubit" in plugins:
        devices_to_test.append("lightning.qubit")
    if "lightning.gpu" in plugins:
        devices_to_test.append("lightning.gpu")
        
    import time
    for dev_name in devices_to_test:
        try:
            # Setup device and circuit
            dev = qml.device(dev_name, wires=4)  # Increase to 4 wires for better timing
            
            @qml.qnode(dev, interface="torch")
            def circuit(phi):
                # Simple layer of rotations and entaglement
                for i in range(4):
                    qml.RY(phi, wires=i)
                for i in range(3):
                    qml.CNOT(wires=[i, i+1])
                return [qml.expval(qml.PauliZ(i)) for i in range(4)]
            
            # Warmup
            phi = torch.tensor(0.1, requires_grad=True)
            _ = circuit(phi)
            
            # Timing
            t0 = time.perf_counter()
            for _ in range(100): # Run 100 times
                res = circuit(phi)
                # Compute gradients to test Torch integration too
                res[0].backward() 
            avg_time = (time.perf_counter() - t0) / 100 * 1000
            
            print(f"{dev_name:<20} | {'✓ OK':<10} | {avg_time:.2f}")
        except Exception as e:
            print(f"{dev_name:<20} | {'❌ FAIL':<10} | N/A")
            # print(f"    Error: {e}") # Uncomment for verbose error

    print("\n" + "="*60)
    print("Diagnostics complete.")

if __name__ == "__main__":
    check_system()
