"""Fetch Esri World Imagery (Web Mercator) for a UTM14N (EPSG:26914) bbox; returns an RGB array."""
import io, requests, numpy as np
from PIL import Image
from pyproj import Transformer
URL = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
_T = Transformer.from_crs(26914, 3857, always_xy=True)
def esri_chip(xmin, ymin, xmax, ymax, res=0.3):
    (x0, x1), (y0, y1) = _T.transform([xmin, xmax], [ymin, ymax])
    w, h = int(round((xmax - xmin) / res)), int(round((ymax - ymin) / res))
    p = dict(bbox=f"{x0},{y0},{x1},{y1}", bboxSR=3857, imageSR=3857, size=f"{w},{h}", format="jpg", f="image")
    r = requests.get(URL, params=p, timeout=120); r.raise_for_status()
    return np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))
def esri_meta(lon, lat):
    """Acquisition metadata (date, resolution, source) of the Esri imagery at a point."""
    u = "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/identify"
    r = requests.get(u, params=dict(geometry=f"{lon},{lat}", geometryType="esriGeometryPoint", sr=4326,
        layers="all", tolerance=1, mapExtent=f"{lon-.01},{lat-.01},{lon+.01},{lat+.01}", imageDisplay="400,400,96",
        returnGeometry="false", f="json"), timeout=60)
    return r.json()
