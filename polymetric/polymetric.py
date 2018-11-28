import numpy as np
import shapely
import shapely.geometry
import shapely.ops
import shapely.affinity
import copy
from enum import Enum


class Shape:
    DEFAULT_PARAMS = {
    }

    def __init__(self, children=[], name=None, **kw):
        # combine all default parameters from subclasses
        # process bottom to top (so higher classes can override params)
        subclasses = self.__class__.__mro__[::-1]
        self._params = {}
        for subcl in subclasses:
            if hasattr(subcl, "DEFAULT_PARAMS"):
                self._params.update(subcl.DEFAULT_PARAMS)
        # finally add any user-supplied params
        self._params.update(kw)

        self.name = name
        if isinstance(children, (tuple, list)):
            self.children = children
        else:
            self.children = [children]
        self.built = False

    def set_params(self, **kw):
        self.built = False
        self._params.update(kw)

    def get_param(self, name, do_conversion=True):
        p = self._params[name]
        if do_conversion and callable(p):
            return p(self.get_param)
        else:
            return p

    def polygonize(self, rebuild=False):
        if not self.built or rebuild:
            self.build()
        return self._polygonize()

    def _polygonize(self):
        raise NotImplementedError("Shapes must override the polygonize-method")

    def build(self):
        self.built = True

    def is_built(self):
        return self.built

    def clone(self, **kw):
        new_instance = copy.deepcopy(self)
        new_instance.set_params(**kw)
        return new_instance

    def apply(self, cls, name="", **kw):
        return cls(children=[self], name=name, **kw)

    def get_exterior_vertex_lists(self):
        polygons = self.apply(Expanded).polygonize()
        vertex_lists = []
        for poly in polygons:
            vertices = list(poly.exterior.coords)
            vertex_lists.append(vertices)

        return vertex_lists

    def get_interior_vertex_lists(self):
        polygons = self.apply(Expanded).polygonize()
        polygons_lists = []
        # iterate over all polygons...
        for poly in polygons:
            vertex_lists = []
            # ..and then interate over all interiors
            for interior in poly.interiors:
                vertices = list(interior.coords)
                vertex_lists.append(vertices)
            polygons_lists.append(vertex_lists)

        return polygons_lists

    def has_interiors(self):
        polygons = self.apply(Expanded).polygonize()
        for poly in polygons:
            if len(poly.interiors) > 0:
                return True

        return False

    def get_bounding_box(self):
        # this is not very efficient as it does the whole polygonization procedure
        # but we're not caring about performance right now
        # (caching polygonizations would be the obvious solution,
        # but not trivial to get right as children might change for example)
        own_poly = self.apply(Flattened).polygonize()[0]

        return own_poly.bounds


class Polygon(Shape):
    DEFAULT_PARAMS = {
        "shell": None,
        "holes": None,
    }

    def _polygonize(self):
        return shapely.geometry.Polygon(**self._params)


class Transformed(Shape):
    DEFAULT_PARAMS = {
        "transformer": lambda get_param, polygon: polygon,
    }

    def _polygonize(self):
        transformer = self.get_param("transformer", do_conversion=False)

        transformed_polys = []
        for c in self.children:
            c_polys = c.polygonize()
            tc_polys = [transformer(self.get_param, c_poly) for c_poly in c_polys]

            transformed_polys.extend(tc_polys)

        return transformed_polys


class Scaled(Transformed):
    DEFAULT_PARAMS = {
        "scales": (1.0, 1.0),
        "origin": (0.0, 0.0),
        "transformer": lambda get_param, polygon: shapely.affinity.scale(
            polygon,
            xfact=get_param("scales")[0],
            yfact=get_param("scales")[1],
            zfact=1.0,  # we don't deal in 3D objects (yet)
            origin=get_param("origin")
        ),
    }


class Rotated(Transformed):
    DEFAULT_PARAMS = {
        "angle": 0.0,
        "origin": (0.0, 0.0),
        "transformer": lambda get_param, polygon: shapely.affinity.rotate(
            polygon,
            angle=get_param("angle"),
            origin=get_param("origin"),
            use_radians=True
        )
    }


class Translated(Transformed):
    DEFAULT_PARAMS = {
        "offset": (0.0, 0.0),
        "dx": lambda get_param: get_param("offset")[0],
        "dy": lambda get_param: get_param("offset")[1],
        "transformer": lambda get_param, polygon: shapely.affinity.translate(
            polygon,
            xoff=get_param("dx"),
            yoff=get_param("dy"),
            zoff=0.0,  # we don't deal in 3D objects (yet)
        )
    }


class Skewed(Transformed):
    DEFAULT_PARAMS = {
        "angles": (0.0, 0.0),
        "origin": (0.0, 0.0),
        "transformer": lambda get_param, polygon: shapely.affinity.skew(
            polygon,
            xs=get_param("angle")[0],
            ys=get_param("angle")[1],
            origin=get_param("origin"),
            use_radians=True
        )
    }


class AffineTransformed(Transformed):
    DEFAULT_PARAMS = {
        "matrix": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "transformer": lambda get_param, polygon: shapely.affinity.affine_transform(
            Polygon,
            get_param("matrix")
        )
    }


class Positioned(Shape):
    DEFAULT_PARAMS = {
        "x": 0.0,
        "y": 0.0,
    }


class Anchors(Enum):
    CENTER = 0
    LOWER_LEFT = 1
    LOWER_RIGHT = 2
    UPPER_RIGHT = 3
    UPPER_LEFT = 4


class AnchorPositioned(Positioned):
    DEFAULT_PARAMS = {
        "anchor": Anchors.CENTER,
    }

    def _polygonize(self):
        # determine bounding box
        bounding_box = Flattened(self.children).get_bounding_box()

        anchor = self.get_param("anchor")

        # determine offsets to position anchor at requested position
        dx = self.get_param("x")
        dy = self.get_param("y")

        if anchor == Anchors.CENTER:
            dx -= (bounding_box[2] + bounding_box[0]) / 2.0
            dy -= (bounding_box[3] + bounding_box[1]) / 2.0
        elif anchor == Anchors.LOWER_LEFT:
            dx -= bounding_box[0]
            dy -= bounding_box[1]
        elif anchor == Anchors.LOWER_RIGHT:
            dx -= bounding_box[2]
            dy -= bounding_box[1]
        elif anchor == Anchors.UPPER_RIGHT:
            dx -= bounding_box[2]
            dy -= bounding_box[3]
        elif anchor == Anchors.UPPER_LEFT:
            dx -= bounding_box[0]
            dy -= bounding_box[3]
        else:
            raise ValueError("Invalid anchor selected: " + anchor)

        # do translation on a fresh copy of children
        # to preserve interior boundaries between the children
        # not the most efficient but I don't care about that right now
        translated_children = Translated(self.children, dx=dx, dy=dy)

        return translated_children.polygonize()


class Ellipse(Positioned):
    DEFAULT_PARAMS = {
        "a": 1.0,
        "b": 1.0,
        "n_sectors": 24,
        "alpha_0": 0.0
    }

    def _polygonize(self):
        angle_list = np.linspace(0, 2.0 * np.pi, self.get_param("n_sectors"), endpoint=False) + self.get_param("alpha_0")

        xs = self.get_param("x") + self.get_param("a") * np.cos(angle_list)
        ys = self.get_param("y") + self.get_param("b") * np.sin(angle_list)

        coords = zip(xs, ys)

        polygon = shapely.geometry.Polygon(coords)

        return [polygon]


class Circle(Ellipse):
    DEFAULT_PARAMS = {
        "r": 1.0,
        "a": lambda get_param: get_param("r"),
        "b": lambda get_param: get_param("r"),
    }


class Rectangle(Positioned):
    DEFAULT_PARAMS = {
        "w": 1.0,
        "h": 1.0,
    }

    def _polygonize(self):
        half_width = np.abs(self.get_param("w")) / 2.0
        half_height = np.abs(self.get_param("h")) / 2.0

        center_x = self.get_param("x")
        center_y = self.get_param("y")

        xs = [center_x - half_width, center_x - half_width, center_x + half_width, center_x + half_width]
        ys = [center_y - half_height, center_y + half_height, center_y + half_height, center_y - half_height]

        coords = zip(xs, ys)

        polygon = shapely.geometry.Polygon(coords)

        return [polygon]


class ParametricSweep(Shape):
    DEFAULT_PARAMS = {
        "constructor": Shape,
        "sweep_over": [],
        "sweep_params": {},
        "fixed_params": {}
    }

    def build(self):
        sweep_over = self.get_param("sweep_over")
        sweep_params = self.get_param("sweep_params")
        fixed_params = self.get_param("fixed_params")
        ctr = self.get_param("constructor", do_conversion=False)

        children = []

        for i in range(len(sweep_over)):
            sweep_index = sweep_over[i]
            swept_params = {}
            for sp_name, sp_value in sweep_params.items():
                if sp_value is True:
                    swept_params[sp_name] = sweep_index
                else:
                    swept_params[sp_name] = sp_value(sweep_index)

            combined_params = {**fixed_params, **swept_params}
            shape = ctr(**combined_params)

            children.append(shape)

        self.children = children
        self.built = True

    def _polygonize(self):
        children_polys = []

        for child in self.children:
            children_polys += child.polygonize()

        return children_polys


class Flattened(Shape):
    def _polygonize(self):
        children_polys = []

        for child in self.children:
            children_polys += child.polygonize()

        single_poly = shapely.ops.cascaded_union(children_polys)

        return [single_poly]


class Expanded(Shape):
    def _polygonize(self):
        children_polys = []

        for child in self.children:
            child_poly = child.polygonize()
            for cp in child_poly:
                # expand multipolygons
                if isinstance(cp, shapely.geometry.MultiPolygon):
                    children_polys += list(cp.geoms)
                else:
                    children_polys += [cp]

        return children_polys


class Combined(Shape):
    def _polygonize(self):
        children_polys = []

        for child in self.children:
            children_polys += child.polygonize()

        return children_polys


class BinaryOperation(Shape):
    DEFAULT_PARAMS = {
        "operation": None
    }

    def _polygonize(self):
        left = self.children[0].apply(Flattened)
        right = self.children[1].apply(Flattened)

        poly_left = left.polygonize()[0]
        poly_right = right.polygonize()[0]

        func = getattr(poly_left, self.get_param("operation"))

        poly_result = func(poly_right)

        return [poly_result]


class Difference(BinaryOperation):
    DEFAULT_PARAMS = {
        "operation": "difference"
    }


class Union(BinaryOperation):
    DEFAULT_PARAMS = {
        "operation": "union"
    }


class Intersection(BinaryOperation):
    DEFAULT_PARAMS = {
        "operation": "intersection"
    }


class SymmetricDifference(BinaryOperation):
    DEFAULT_PARAMS = {
        "operation": "symmetric_difference"
    }
