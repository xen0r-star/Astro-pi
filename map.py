from PIL import Image, ImageDraw
import math
from pathlib import Path

class GlobalMercator:
    def __init__(self):
        # Constants
        self.equatorial_radius = 6378137
        self.tile_size = 256
        self.initial_resolution = math.pi * 2 * self.equatorial_radius / self.tile_size
        self.origin_shift = math.pi * 2 * self.equatorial_radius / 2
        self.zoom = 3

    def convert_lat_long_to_meters(self, lat, lon):
        m_x = lon * self.origin_shift / 180
        m_y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)
        m_y = m_y * self.origin_shift / 180
        return {'mX': m_x, 'mY': m_y}

    def convert_lat_long_to_pixels(self, lat, lon):
        meters = self.convert_lat_long_to_meters(lat, lon)
        pixels = self.convert_meters_to_pixels(meters['mX'], meters['mY'])
        return pixels

    def convert_meters_to_pixels(self, m_x, m_y):
        res = self.calc_resolution()
        p_x = (m_x + self.origin_shift) / res
        p_y = (m_y + self.origin_shift) / res
        return {'pX': p_x, 'pY': p_y}

    def calc_resolution(self):
        calc_val = self.initial_resolution / math.pow(2, self.zoom)
        return calc_val


def draw_point_on_map(image_path, lat, lon, output_path):
    world_map = Image.open(image_path)

    mercator = GlobalMercator()
    pixels = mercator.convert_lat_long_to_pixels(latitude, longitude)
    x = pixels['pX']
    y = pixels['pY']

    draw = ImageDraw.Draw(world_map)
    point_size = 3
    draw.ellipse([x - point_size, y - point_size, x + point_size, y + point_size], fill="#b52b10", outline="#760000")

    world_map.save(output_path)


paths = Path(__file__).parent.resolve()

latitude = 0
longitude = 0

world_map_path = paths / "Map.png"
output_image_path = paths / "Map2.png"


draw_point_on_map(world_map_path, latitude, longitude, output_image_path)
