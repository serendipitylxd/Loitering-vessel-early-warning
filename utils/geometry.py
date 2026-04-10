import math
from typing import List, Tuple

Point = Tuple[float, float]

def box2d_corners_xy(x: float, y: float, dx: float, dy: float, yaw: float) -> List[Point]:
    """Return the four corners of an oriented rectangle on the XY plane (counter-clockwise), considering yaw rotation.
    (dx, dy) are the side lengths along local x and y directions. yaw rotates around z-axis (right-hand rule)."""
    hx, hy = dx * 0.5, dy * 0.5
    # Local vertices (counter-clockwise): +x+y, -x+y, -x-y, +x-y
    local = [( hx,  hy), (-hx,  hy), (-hx, -hy), ( hx, -hy)]
    c, s = math.cos(yaw), math.sin(yaw)
    world = []
    for lx, ly in local:
        wx = x + c * lx - s * ly
        wy = y + s * lx + c * ly
        world.append((wx, wy))
    return world

def segments_of_poly(poly: List[Point]):
    n = len(poly)
    for i in range(n):
        yield poly[i], poly[(i + 1) % n]

def seg_intersect_horizontal(seg: Tuple[Point, Point], y0: float, xr: Tuple[float, float]) -> bool:
    """Check if a line segment intersects with a horizontal line y=y0 within x∈xr (including collinear overlaps)."""
    (x1,y1), (x2,y2) = seg
    x_min, x_max = min(x1,x2), max(x1,x2)
    xr0, xr1 = xr

    # Collinear case
    if abs(y1 - y0) < 1e-9 and abs(y2 - y0) < 1e-9:
        # Check if it overlaps with [xr0, xr1]
        return not (x_max < xr0 or x_min > xr1)

    # Crosses y0
    if (y1 - y0) * (y2 - y0) < 0:
        t = (y0 - y1) / (y2 - y1)
        x_at = x1 + t * (x2 - x1)
        return (xr0 - 1e-9) <= x_at <= (xr1 + 1e-9)
    return False

def poly_intersect_stop_line(poly: List[Point], stop_y: float, xr: Tuple[float,float]) -> bool:
    for seg in segments_of_poly(poly):
        if seg_intersect_horizontal(seg, stop_y, xr):
            return True
    return False

def poly_overlaps_band(poly: List[Point], xr: Tuple[float,float], yr: Tuple[float,float]) -> bool:
    """Check if a polygon overlaps with the rectangular band [xr0,xr1] × [yr0,yr1] 
    (coarse check: separating axis approximation)."""
    xr0, xr1 = xr
    yr0, yr1 = yr
    # 1) Any vertex inside the band
    for x,y in poly:
        if xr0 <= x <= xr1 and yr0 <= y <= yr1:
            return True
    # 2) Check polygon edges against rectangle edges (conservative)
    rect = [(xr0,yr0),(xr1,yr0),(xr1,yr1),(xr0,yr1)]
    rect_edges = list(segments_of_poly(rect))
    for e1 in segments_of_poly(poly):
        for e2 in rect_edges:
            if segments_intersect(e1, e2):
                return True
    return False

def ccw(a:Point,b:Point,c:Point)->float:
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])

def on_segment(a:Point,b:Point,p:Point)->bool:
    return min(a[0],b[0])-1e-9 <= p[0] <= max(a[0],b[0])+1e-9 and \
           min(a[1],b[1])-1e-9 <= p[1] <= max(a[1],b[1])+1e-9

def segments_intersect(seg1, seg2)->bool:
    a,b = seg1; c,d = seg2
    d1 = ccw(a,b,c); d2 = ccw(a,b,d)
    d3 = ccw(c,d,a); d4 = ccw(c,d,b)
    if (d1==0 and on_segment(a,b,c)) or (d2==0 and on_segment(a,b,d)) or \
       (d3==0 and on_segment(c,d,a)) or (d4==0 and on_segment(c,d,b)):
        return True
    return (d1>0) != (d2>0) and (d3>0) != (d4>0)




