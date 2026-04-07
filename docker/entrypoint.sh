#!/bin/bash
# OpenEnv evaluation entrypoint
# Runs the programmatic grader and outputs JSON results
mkdir -p /results
python grade.py --episodes 20 --output /results/grade_output.json
