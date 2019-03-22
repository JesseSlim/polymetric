import gdspy
from importlib import reload


# should we reload the gdspy module every time something is saved?
DO_RELOAD = True

# maximum number of vertices per polygon. Absolute limit set by GDS file format is 8191
# can be adjusted by user
MAX_VERTICES = 8000


def save(shape, filename, cell_name="POLYGON", datatype=1000, ignore_interiors=False, **kw):
    if DO_RELOAD:
        reload(gdspy)
    poly_cell = gdspy.Cell(cell_name)
    if not ignore_interiors and shape.has_interiors():
        raise ValueError("Polygon contains interiors, which can not be represented in the GDS file format. To ignore, call with ignore_interiors=True")

    vertex_lists = shape.get_exterior_vertex_lists()

    for vl in vertex_lists:
        if len(vl) > MAX_VERTICES:
            raise ValueError("Polygon contains %d vertices, more than the limit of %d vertices" % (len(vl), MAX_VERTICES))
        
        # shapely always duplicates the first vertex as the last vertex, get rid of that
        gds_poly = gdspy.Polygon(vl[:-1], datatype=datatype)
        poly_cell.add(gds_poly)

    gdspy.write_gds(filename, **kw)

def save_multiple(shapes, filename, layers=None, cell_name="POLYGON", datatype=1000, ignore_interiors=False, **kw):
    if DO_RELOAD:
        reload(gdspy)
    poly_cell = gdspy.Cell(cell_name)
    for i, shape in enumerate(shapes):
        if not ignore_interiors and shape.has_interiors():
            raise ValueError("Polygon contains interiors, which can not be represented in the GDS file format. To ignore, call with ignore_interiors=True")
 
        vertex_lists = shape.get_exterior_vertex_lists()
 
        for vl in vertex_lists:
            if len(vl) > MAX_VERTICES:
                raise ValueError("Polygon contains %d vertices, more than the limit of %d vertices" % (len(vl), MAX_VERTICES))
 
            layer = 0
            if layers is not None:
                layer = layers[i]
            # shapely always duplicates the first vertex as the last vertex, get rid of that
            gds_poly = gdspy.Polygon(vl[:-1], datatype=datatype, layer=layer)
            poly_cell.add(gds_poly)
           
    gdspy.write_gds(filename, **kw)
        