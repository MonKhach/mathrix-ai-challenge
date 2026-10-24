"""Memory report for TESTS.md B3 (does everything fit in 8 GB during a call?) and the fp32/int8
decision (D17). Windows only: uses the peak working set, which Windows keeps per process."""
import psutil

MB = 2**20
GB = 2**30


def memory_line(moment: str) -> str:
    """One line: this program's RAM now and its peak, plus RAM still available on the whole PC
    (which includes the call app and everything else that is running)."""
    process = psutil.Process().memory_info()
    system = psutil.virtual_memory()
    return (f"[memory, {moment}] program {process.rss / MB:.0f} MB now, "
            f"peak {process.peak_wset / MB:.0f} MB | PC: {system.available / GB:.1f} GB available "
            f"of {system.total / GB:.1f} GB ({system.percent:.0f} % used)")
