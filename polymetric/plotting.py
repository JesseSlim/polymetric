import numpy as np
import matplotlib.pyplot as plt

import shapely as sp
import shapely.geometry


def plot_polys(polys, cycle_colours=False, **kw):
    single_color = None
    for i, outer_poly in enumerate(polys):
        if isinstance(outer_poly, (sp.geometry.MultiPolygon)):
            inner_poly = outer_poly.geoms
        else:
            inner_poly = [outer_poly]
        
        for poly in inner_poly:
            xs, ys = poly.exterior.xy
            plot_params = dict(kw)

            if cycle_colours:
                plot_params["color"] = "C%d" % (i % 10)
            elif single_color is not None:
                plot_params["color"] = single_color

            l = plt.plot(xs, ys, **plot_params)
            if single_color is None:
                single_color = l[0].get_color()
                plot_params["color"] = single_color

            # plot interiors with a different line style
            plot_params["linestyle"] = "--"
            for interior in poly.interiors:
                xs, ys = interior.xy
                plt.plot(xs, ys, **plot_params)


def plot_shapes(shapes, *more_shapes, **kw):
    if not isinstance(shapes, (list, tuple)):
        shapes = [shapes]
        
    shapes += more_shapes
    for s in shapes:
        plot_polys(s.polygonize(), **kw)


def show_polys():
    plt.gca().set_aspect('equal')
    plt.show()
    

def inspect_shapes(shapes, *more_shapes, figure=None, **kw):
    if figure is None:
        plt.figure()
    plot_shapes(shapes, *more_shapes, **kw)
    show_polys()