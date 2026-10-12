"""Fetch Waveshare's RoArm-M2 link meshes (MIT) into www/models/, at the revision
roarm-haptic-bridge verified. Without them the page draws the arm from simple
shapes. Usage: python fetch_models.py"""
from pathlib import Path
from urllib.request import urlopen

REVISION = '40dbd84b553695212fab713e8465f817ba95454d'
BASE_URL = (
    'https://raw.githubusercontent.com/waveshareteam/roarm_ws/'
    f'{REVISION}/src/roarm_main/roarm_description/meshes/roarm_m2/'
)
FILES = ['base_link.stl', 'link1.stl', 'link2.stl', 'link3.stl', 'gripper_link.stl']
DESTINATION = Path(__file__).parent / 'www' / 'models'

DESTINATION.mkdir(exist_ok=True)
for name in FILES:
    print(f'Fetching {name}')
    with urlopen(BASE_URL + name, timeout=30) as res:
        (DESTINATION / name).write_bytes(res.read())
print(f'Saved to {DESTINATION}')
