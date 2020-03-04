import gdspy
from importlib import reload
from .. import polymetric as pm


# should we reload the gdspy module every time something is saved/loaded?
# GDSPY keeps some global library of stuff in memory that might need to be wiped
DO_RELOAD = True

# maximum number of vertices per polygon. Absolute limit set by GDS file format is 8191
# can be adjusted by user
MAX_VERTICES = 8000


def save(shape, filename, cell_name="POLYGON", datatype=1000,
         ignore_interiors=False, **kw):
    if DO_RELOAD:
        reload(gdspy)
    poly_cell = gdspy.Cell(cell_name)
    if not ignore_interiors and shape.has_interiors():
        raise ValueError(
            "Polygon contains interiors, which can not be represented"
            " in the GDS file format. To ignore, call with ignore_interiors=True")

    vertex_lists = shape.get_exterior_vertex_lists()

    for vl in vertex_lists:
        if len(vl) > MAX_VERTICES:
            raise ValueError(f"Polygon contains {len(vl)} vertices, more than the limit of {MAX_VERTICES} vertices")

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
            raise ValueError(
                "Polygon contains interiors, which can not be represented"
                " in the GDS file format. To ignore, call with ignore_interiors=True")

        vertex_lists = shape.get_exterior_vertex_lists()

        for vl in vertex_lists:
            if len(vl) > MAX_VERTICES:
                raise ValueError("Polygon contains {} vertices, more than the"
                    " limit of {} vertices".format(len(vl), MAX_VERTICES))

            layer = 0
            if layers is not None:
                layer = layers[i]
            # shapely always duplicates the first vertex as the last vertex, get rid of that
            gds_poly = gdspy.Polygon(vl[:-1], datatype=datatype, layer=layer)
            poly_cell.add(gds_poly)

    gdspy.write_gds(filename, **kw)


def load(filename):
    if DO_RELOAD:
        reload(gdspy)

    gdslib = gdspy.GdsLibrary(infile=filename)

    polys_per_layer = {}

    for _, cell in gdslib.cell_dict.items():
        for p1 in cell.polygons:
            for (layer, poly) in zip(p1.layers, p1.polygons):
                if layer not in polys_per_layer:
                    polys_per_layer[layer] = []

                polys_per_layer[layer].append(pm.Polygon(shell=poly))

    return polys_per_layer