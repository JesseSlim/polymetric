import ezdxf


def save(shape, filename, ignore_interiors=False, dxf_version="R2010", **kw):
    if not ignore_interiors and shape.has_interiors():
        raise ValueError("Polygon contains interiors, which this saving routine can not handle. To ignore, call with ignore_interiors=True")

    dwg = ezdxf.new(dxf_version)
    msp = dwg.modelspace()

    vertex_lists = shape.get_exterior_vertex_lists()

    for vl in vertex_lists:
        # shapely always duplicates the first vertex as the last vertex, get rid of that
        polyline = msp.add_lwpolyline(vl[:-1])
        polyline.closed = True

    dwg.saveas(filename, **kw)