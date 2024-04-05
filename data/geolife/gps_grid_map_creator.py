import numpy as np

class GPSGridMapCreator():
    def __init__(self, grid_size_meter=1):
        self.grid_size_meter = grid_size_meter
        self.lat1 = 0
        self.lon1 = 0
        self.grid_numbers = 0
        self.lat_degrees = 0
        self.long_degrees = 0
        self.num_lat = 0
        self.num_lon = 0
        self.latitudes = []
        self.longitudes = []

    def get_num_lat(self):
        return self.num_lat # row
    def get_num_lon(self):
        return self.num_lon # col
    def get_num_lat_lon(self):
        return self.num_lat, self.num_lon

    def km_to_degrees(self, latitude, kilometers):
        # Earth's radius in kilometers
        earth_radius_km = 6371.0

        # Convert kilometers to radians
        angle_rad = kilometers / earth_radius_km

        # Convert radians to degrees
        angle_deg = np.degrees(angle_rad)

        # Correction factor for latitude
        lat_correction = np.cos(np.radians(latitude))

        # Convert degrees to adjusted degrees
        adjusted_degrees = angle_deg / lat_correction

        return adjusted_degrees

    def meter_to_degrees(self, latitude, meters):
        # Convert meters to kilometers
        kilometers = meters / 1000

        # Convert kilometers to degrees using the km_to_degrees function
        degrees = self.km_to_degrees(latitude, kilometers)

        return degrees

    def create_grid_map(self, min_lat, min_lon, max_lat, max_lon):
        lat1, lon1 = min_lat, min_lon
        lat2, lon2 = max_lat, max_lon
        print(f"lat1: {lat1}, lon1: {lon1}, lat2: {lat2}, lon2: {lon2}")

        self.lat1 = lat1
        self.lon1 = lon1
        # Convert grid size from meters to degrees
        self.lat_degrees = self.meter_to_degrees((lat1 + lat2) / 2, self.grid_size_meter)
        self.lon_degrees = self.meter_to_degrees((lon1 + lon2) / 2, self.grid_size_meter)

        # print(f"lat_degrees: {self.lat_degrees}, lon_degree: {self.lon_degrees}")
        # Calculate the number of grid points in latitude and longitude directions
        self.num_lat = int(np.abs(lat2 - lat1) / np.abs(self.lat_degrees))
        self.num_lon = int(np.abs(lon2 - lon1) / np.abs(self.lon_degrees))
        # print(f"lon2 - lon1: {np.abs(lon2 - lon1)}")
        print(f"len of lat: {self.num_lat}, len of lon: {self.num_lon}")

        # Generate latitude and longitude grid points
        # self.latitudes = np.linspace(lat1, lat2, self.num_lat)
        # self.longitudes = np.linspace(lon1, lon2, self.num_lon)

        # Create a 2D grid for numbering
        self.grid_numbers = (self.num_lat + 1) * (self.num_lon + 1)
        print(f"grid_shape: ({(self.num_lat + 1)}, {(self.num_lon + 1)})")
        print(f"gird_number: {self.grid_numbers}")

    def find_grid_number(self, lat, lon):
        # grid_lat = int(np.abs(lat - self.lat1) / np.abs(self.lat_degrees))
        # grid_lon = int(np.abs(lon - self.lon1) / np.abs(self.lon_degrees))
        grid_row = np.abs(lat - self.lat1) / np.abs(self.lat_degrees)
        grid_col = np.abs(lon - self.lon1) / np.abs(self.lon_degrees)
        grid_row, grid_col = self.get_normalization_from_grid(grid_row, grid_col)

        # grid_lat = self.lat1 + (np.abs(self.lat_degrees) * grid_row)
        # grid_lon = self.lon1 + (np.abs(self.lon_degrees) * grid_col)

        # grid_num = grid_row * (self.num_lon + 1) + grid_col + 1
        # print(f"grid_lat: { grid_lat * (self.num_lon + 1)}")
        # print(f"grid_lat: {grid_row}, gird_lon: {grid_col}")
        # return grid_row, grid_col, grid_lat, grid_lon, grid_num
        return grid_row, grid_col#, grid_lat, grid_lon, grid_num

    def get_normalization_from_grid(self, grid_row, grid_col):
        normal_row = grid_row / self.num_lat;
        normal_col = grid_col / self.num_lon;
        return normal_row, normal_col

    def get_grid_form_normalization(self, normal_row, normal_col):
        grid_row = normal_row * self.num_lat;
        grid_col = normal_col * self.num_lon;
        return grid_row, grid_col

# Example coordinates
# lat1, lon1 = 37.0, -122.0  # Lower-left corner
# lat2, lon2 = 38.0, -121.0  # Upper-right corner
# grid_size_meter = 1000  # Size of each grid in meters

# mapCreator = GPSGridMapCreator(grid_size_meter)
# mapCreator.create_grid_map(lat1, lon1, lat2, lon2)
# print(mapCreator.find_grid_number(37.3, -121.8))
# print("done")