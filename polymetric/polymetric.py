import numpy as np
import shapely
import shapely.geometry
import shapely.ops
import shapely.affinity
import copy

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
        self.children = children
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

    def clone(self):
        return copy.deepcopy(self)

    def apply(self, cls, name="", **kw):
        return cls(children=[self], name=name, **kw)

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
            yfact=get_param("scales")[0],
            zfact=1.0, # we don't deal in 3D objects (yet)
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
        "transformer": lambda get_param, polygon: shapely.affinity.translate(
            polygon,
            xoff=get_param("offset")[0],
            yoff=get_param("offset")[1],
            zoff=0.0, # we don't deal in 3D objects (yet)
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

class Ellipse(Positioned):
    DEFAULT_PARAMS = {
        "a": 1.0,
        "b": 1.0,
        "n_sectors": 24,
        "alpha_0": 0.0
    }

    def _polygonize(self):
        angle_list = np.linspace(0, 2*np.pi, self.get_param("n_sectors"), endpoint=False) + self.get_param("alpha_0")

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
        width = np.abs(self.get_param("w"))
        height = np.abs(self.get_param("h"))

        center_x = self.get_param("x")
        center_y = self.get_param("y")

        xs = [center_x - width, center_x - width, center_x + width, center_x + width]
        ys = [center_y - height, center_y + height, center_y + height, center_y - height]

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

        # polygon = shapely.ops.cascaded_union(children_polys)

        return children_polys

class Flattened(Shape):
    def _polygonize(self):
        children_polys = []

        for child in self.children:
            children_polys += child.polygonize()

        single_poly = shapely.ops.cascaded_union(children_polys)

        return [single_poly]

class BinaryOperation(Shape):
    DEFAULT_PARAMS = {
        "operation": None
    }

    def _polygonize(self):
        left = Flattened(children=[self.children[0]])
        right = Flattened(children=[self.children[1]])

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