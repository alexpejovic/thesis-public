import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parent2_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent2_dir)

from simple_trace import plot_simple_trace

if __name__ == "__main__":
    plot_simple_trace("ideal_trace.svg", 1201, 5)
