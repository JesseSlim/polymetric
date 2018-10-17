import numpy as np
import shapely as sp
import shapely.geometry
import shapely.ops

class Shape:
    DEFAULT_PARAMS = {
    }
    def __init__(self, name=None, **kw):
        self._params = {**Shape.DEFAULT_PARAMS, **kw}
        self.name = name
        self.children = []
        self.built = False

    def set_params(self, **kw):
        self._params.update(kw)

    def get_param(self, name, do_conversion=True):
        p = self._params[name]
        if do_conversion and callable(p):
            return p(self._params)
        else:
            return p

    def polygonize(self):
        raise NotImplementedError("Shapes must override the polygonize-method")

    def build(self):
        self.built = True

    def is_built(self):
        return self.built

class PositionedShape(Shape):
    DEFAULT_PARAMS = {
        "x": 0.0,
        "y": 0.0,
    }
    def __init__(self, name=None, **kw):
        combined_params = {**PositionedShape.DEFAULT_PARAMS, **kw}
        super().__init__(name=name, **combined_params)

class Ellipse(PositionedShape):
    DEFAULT_PARAMS = {
        "a": 1.0,
        "b": 1.0,
        "n_sectors": 24,
    }

    def __init__(self, name=None, **kw):
        combined_params = {**Ellipse.DEFAULT_PARAMS, **kw}
        super().__init__(name=name, **combined_params)

    def polygonize(self):
        self.build()

        angle_list = np.linspace(0, 2*np.pi, self.get_param("n_sectors"), endpoint=False)

        xs = self.get_param("x") + self.get_param("a") * np.cos(angle_list)
        ys = self.get_param("y") + self.get_param("b") * np.sin(angle_list)

        coords = zip(xs, ys)

        polygon = sp.geometry.Polygon(coords)

        return polygon

class Circle(Ellipse):
    DEFAULT_PARAMS = {
        "r": 1.0,
        "a": lambda params: params["r"],
        "b": lambda params: params["r"],
    }
    
    def __init__(self, name=None, **kw):
        combined_params = {**Circle.DEFAULT_PARAMS, **kw}
        super().__init__(name=name, **combined_params)

class Rectangle(PositionedShape):
    DEFAULT_PARAMS = {
        "w": 1.0,
        "h": 1.0,
    }

    def __init__(self, name=None, **kw):
        combined_params = {**Rectangle.DEFAULT_PARAMS, **kw}
        super().__init__(name=name, **combined_params)

    def polygonize(self):
        self.build()

        width = np.abs(self.get_param("w"))
        height = np.abs(self.get_param("h"))

        center_x = self.get_param("x")
        center_y = self.get_param("y")

        xs = [center_x - width, center_x - width, center_x + width, center_x + width]
        ys = [center_y - height, center_y + height, center_y + height, center_y - height]

        coords = zip(xs, ys)

        polygon = sp.geometry.Polygon(coords)

        return polygon

class ParametricSweep(Shape):
    DEFAULT_PARAMS = {
        "constructor": Shape,
        "sweep_over": [],
        "sweep_params": {},
        "fixed_params": {}
    }

    def __init__(self, name=None, **kw):
        combined_params = {**ParametricSweep.DEFAULT_PARAMS, **kw}
        super().__init__(name=name, **combined_params)

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
    
    def polygonize(self):
        self.build()
        children_polys = []

        for child in self.children:
            children_polys.append(child.polygonize())

        polygon = sp.ops.cascaded_union(children_polys)

        return polygon