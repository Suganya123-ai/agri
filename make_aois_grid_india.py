import json
import math

def make_grid_around_center(center_lat, center_lon, n=30, box_h=0.02, box_w=0.02, spacing=0.03, prefix="in"):
    """
    Creates ~n AOIs as small bboxes around a center point.
    box_h/box_w in degrees.
    spacing controls distance between AOIs (degrees).
    """
    # grid size ~ sqrt(n)
    k = int(math.ceil(math.sqrt(n)))
    half = k // 2

    aois = []
    idx = 1
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            if len(aois) >= n:
                break

            lat = center_lat + i * spacing
            lon = center_lon + j * spacing

            west = lon
            south = lat
            east = lon + box_w
            north = lat + box_h

            aois.append({
                "aoi_id": f"{prefix}_{idx}",
                "bbox": [west, south, east, north]
            })
            idx += 1

    return aois

if __name__ == "__main__":
    # ✅ CHANGE THIS to your study region in India:
    # Example: Nagpur area (Maharashtra)
    center_lat = 13.00
    center_lon = 77.70
    
    
    aois = make_grid_around_center(
        center_lat=center_lat,
        center_lon=center_lon,
        n=30,
        box_h=0.02,
        box_w=0.02,
        spacing=0.03,
        prefix="india"
    )

    with open("aois.json", "w") as f:
        json.dump(aois, f, indent=2)

    print(f"Saved aois.json with {len(aois)} AOIs around ({center_lat}, {center_lon})")
