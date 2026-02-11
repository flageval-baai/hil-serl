#!/usr/bin/env python3
"""
Simple inference script for running a policy on the Franka robot.

Example usage:
    python -m inference.run_inference --prompt "Pick up the block"
    python -m inference.run_inference --prompt "Put the object in the basket" --hz 15 --max-steps 500
"""

from inference.inference_loop import main

if __name__ == "__main__":
    main()
