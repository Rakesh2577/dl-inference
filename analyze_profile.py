import json

profile_file = "onnxruntime_profile__2026-10-02_12-06-42_473.json"

with open(profile_file, "r") as f:
    events = json.load(f)

operator_times = []

for event in events:
    if event.get("cat") == "Node" and "dur" in event:
        name = event.get("name", "Unknown")
        duration_us = event["dur"]

        operator_times.append((duration_us, name))

operator_times.sort(reverse=True)

print("Top 10 slowest operators:\n")

for duration_us, name in operator_times[:10]:
    print(f"{duration_us / 1000:.3f} ms  ->  {name}")
