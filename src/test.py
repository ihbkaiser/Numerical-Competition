from gnbg_func.GNBG import GNBG
from MSHO import MSHO
import numpy as np
import csv
import os

if __name__ == "__main__":
    DIM = 30
    MAX_FES = 500000
    RUNS_PER_PROB = 31
    LOG_FILE = "msho_run_results_2.csv"

    # Tham số chung cho MSHO
    msho_params = {
        "BASE_POPSZ": 100,
        "learning": True,
        "dynamic_pop": True,
        "phasethree": True,
        "new_LS": False
    }

    # Mở file append và ghi header nếu cần
    write_header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["prob_idx", "run_id", "val", "aceps", "x"])

        # Vòng lặp chính: problem 2–24, mỗi problem chạy RUNS_PER_PROB lần
        for prob_idx in range(1, 25):
            for run_id in range(1, RUNS_PER_PROB + 1):
                print(f"Problem {prob_idx}, run {run_id}/{RUNS_PER_PROB}")
                prob = GNBG(prob_idx)
                algo = MSHO(
                    name=f"MSHO-LLM-p{prob_idx}-run{run_id}",
                    prob=prob,
                    gen_length=DIM,
                    MAX_FES=MAX_FES,
                    **msho_params
                )
                val, aceps, x = algo.run()

                # Chuyển x (numpy array) thành chuỗi
                # Nếu x có dạng (d,1) thì flatten về (d,)
                x_flat = x.flatten() if hasattr(x, "flatten") else np.array(x)
                x_str = " ".join(f"{xi:.6f}" for xi in x_flat)

                # Ghi ngay kết quả của lần chạy này, bao gồm cả x
                writer.writerow([prob_idx, run_id,
                                 f"{val:.6f}", f"{aceps:.6f}", x_str])
                f.flush()  # đảm bảo ghi ngay ra đĩa

                print(f">>> Logged: prob={prob_idx}, run={run_id}, "
                      f"val={val:.6e}, aceps={aceps:.6f}, x=<...>")

    print(f"Đã lưu kết quả mỗi run vào {LOG_FILE}")
