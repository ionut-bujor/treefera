import geopandas as gpd

farms = gpd.read_file("F1-coffee-leaf-rust-ethiopia/F1_CLR Survey Farm Level.geojson")

print("Columns:", farms.columns.tolist())
print("\nShape:", farms.shape)
print("\nFirst few rows:")
print(farms.head())
print("\nDescribe:")
print(farms.describe())
print("\nCRS:", farms.crs)
print("\nBounds:", farms.total_bounds)