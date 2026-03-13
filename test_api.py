import requests
import json
import time

API = "http://localhost:4735"

try:
    meas = requests.get(f"{API}/measurements").json()
    if meas:
        test_id = list(meas.keys())[0]
        
        test_strings = ["ERB", "1/6", "1/6 octave", "Psychoacoustic"]
        
        for s in test_strings:
            req = {
                "processName": "Smooth",
                "measurementIndices": [int(test_id)],
                "parameters": {"smoothing": s}
            }
            res = requests.post(f"{API}/measurements/process-measurements", json=req)
            print(f"Testing '{s}': Status {res.status_code}, Response: {res.text}")
            time.sleep(0.5)
            
except Exception as e:
    print(e)
