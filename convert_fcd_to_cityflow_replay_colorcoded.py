import xml.etree.ElementTree as ET
import math
import sys

# Colors as R,G,B floats in 0..1 (CityFlow frontend convention)
FAKE_COLOR = "1 0 0"      # red for injected/fake vehicles
REAL_COLOR = "1 1 0"      # yellow for real vehicles (change if you prefer)
FAKE_PREFIX = "fake_"     # matches SDSMInjector.get_fake_vehicle_id()


def convert(fcd_xml_path, output_txt_path, default_length=5.0, default_width=2.0):
    timesteps_written = 0
    with open(output_txt_path, 'w') as out:
        try:
            for event, elem in ET.iterparse(fcd_xml_path, events=('end',)):
                if elem.tag != 'timestep':
                    continue

                vehicle_records = []
                for v in elem.findall('vehicle'):
                    x = float(v.attrib['x'])
                    y = float(v.attrib['y'])
                    angle_deg = float(v.attrib['angle'])
                    angle_rad = math.radians(90.0 - angle_deg)
                    while angle_rad < 0:
                        angle_rad += 2 * math.pi
                    while angle_rad >= 2 * math.pi:
                        angle_rad -= 2 * math.pi

                    vid = v.attrib['id']

                    # Color fake vehicles red, real vehicles the default color
                    color = FAKE_COLOR if vid.startswith(FAKE_PREFIX) else REAL_COLOR

                    status = "0 0"
                    vehicle_records.append(
                        f"{x} {y} {angle_rad} {vid} {status} {default_length} {default_width} {color}"
                    )

                tls_str = ""
                line = ",".join(vehicle_records) + ";" + tls_str
                out.write(line + "\n")
                timesteps_written += 1

                # Free memory as we go
                elem.clear()
        except ET.ParseError as e:
            print(f"Warning: parse error at end of file (likely truncated): {e}")
            print(f"Successfully wrote {timesteps_written} timesteps before the error.")

    print(f"Converted {fcd_xml_path} -> {output_txt_path} ({timesteps_written} timesteps)")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python convert_fcd_to_cityflow_replay.py <fcd.xml> <output.txt>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
