# -*- coding: utf-8 -*-

from collections.abc import Iterable
import copy
from enum import Enum

import numpy as np
import shapely
import shapely.geometry
import shapely.ops
import shapely.affinity
import shapely.wkt


class Shape:
    DEFAULT_PARAMS = {}
    PARAMS_EXCLUDED_FROM_REPR = []

    def __init__(self, children=[], name=None, **kw):
        # combine all default parameters from subclasses
        # process bottom to top (so higher classes can override params)
        subclasses = self.__class__.__mro__[::-1]
        self._params = {}
        self._params_excluded_from_repr = set()
        for subcl in subclasses:
            if hasattr(subcl, "DEFAULT_PARAMS"):
                self._params.update(subcl.DEFAULT_PARAMS)

            if hasattr(subcl, "PARAMS_EXCLUDED_FROM_REPR"):
                self._params_excluded_from_repr.update(subcl.PARAMS_EXCLUDED_FROM_REPR)
        # finally add any user-supplied params
        self._params.update(kw)

        self.name = name
        self.children = children if isinstance(children, Iterable) else [children]
        self.built = False

    def set_params(self, **kw):
        self.built = False
        self._params.update(kw)

    def get_param(self, name, do_conversion=True):
        p = self._params[name]
        return p(self.get_param) if do_conversion and callable(p) else p

    def get_params(self, *names, do_conversion=True):
        return [self.get_param(pname, do_conversion) for pname in names]

    def get_all_params(self, do_conversion=True):
        result_params = {}

        for k in self._params:
            result_params[k] = self.get_param(k, do_conversion=do_conversion)

        return result_params

    def polygonize(self, rebuild=False):
        if not self.built or rebuild:
            self.build()
        return self._polygonize()

    def _polygonize(self):
        raise NotImplementedError(
            "Shapes must override the polygonize-method.")

    def build(self):
        self.built = True

    def is_built(self):
        return self.built

    def clone(self, **kw):
        new_instance = copy.deepcopy(self)
        new_instance.set_params(**kw)
        return new_instance

    def apply(self, cls, name=None, **kw):
        if isinstance(cls, Shape):
            # cls is an instantiated shape, copy it and apply it to self
            new_shape = cls.clone(**kw)
            new_shape.children = [self]
            new_shape.name = name
            return new_shape
        else:
            # cls is a class, we need to instantiate it
            return cls(children=[self], name=name, **kw)

    def get_exterior_vertex_lists(self):
        polygons = self.apply(Expanded).polygonize()
        vertex_lists = [list(poly.exterior.coords) for poly in polygons]
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

    def __repr__(self):
        child_reps = [repr(child) for child in self.children]
        class_name = type(self).__name__

        # # we prepend the list with an empty string, to make sure that the param string, if there are any params,
        # # starts with a ', '. This way we can immediately insert it after the other constructor arguments
        # params_as_arguments = ['']
        # for k, v in self._params.items():
        #     if k in self._params_excluded_from_repr:
        #         continue
        #
        #     # let's try our best to get the defining source code in case we're dealing with a lambda

        param_string = ", ".join(
            # we prepend the list with an empty string, to make sure that the param string, if there are any params,
            # starts with a ', '. This way we can immediately insert it after the other constructor arguments
            [''] + [f"{k}={repr(v)}" for k, v in self._params.items() if k not in self._params_excluded_from_repr]
        )

        full_repr = f"{class_name}(children={repr(child_reps)}, name={repr(self.name)}{param_string})"
        return full_repr


class EmptyShape(Shape):
    def _polygonize(self):
        return [shapely.geometry.Polygon()]


class Polygon(Shape):
    DEFAULT_PARAMS = {
        "shell": None,
        "holes": None,
    }

    def _polygonize(self):
        return [shapely.geometry.Polygon(**self.get_all_params())]


class Transformed(Shape):
    DEFAULT_PARAMS = {
        "transformer": lambda get_param, polygon: polygon,
    }
    PARAMS_EXCLUDED_FROM_REPR = ['transformer']

    def _polygonize(self):
        transformer = self.get_param("transformer", do_conversion=False)

        transformed_polys = []
        for c in self.children:
            c_polys = c.polygonize()
            tc_polys = [transformer(self.get_param, c_poly)
                        for c_poly in c_polys]
            transformed_polys += tc_polys

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
    LOWER_CENTER = 5
    RIGHT_CENTER = 6
    UPPER_CENTER = 7
    LEFT_CENTER = 8


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
        elif anchor == Anchors.LOWER_CENTER:
            dx -= (bounding_box[2] + bounding_box[0]) / 2.0
            dy -= bounding_box[1]
        elif anchor == Anchors.RIGHT_CENTER:
            dx -= bounding_box[2]
            dy -= (bounding_box[3] + bounding_box[1]) / 2.0
        elif anchor == Anchors.UPPER_CENTER:
            dx -= (bounding_box[2] + bounding_box[0]) / 2.0
            dy -= bounding_box[3]
        elif anchor == Anchors.LEFT_CENTER:
            dx -= bounding_box[0]
            dy -= (bounding_box[3] + bounding_box[1]) / 2.0
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
        n = self.get_param("n_sectors")
        alpha_0 = self.get_param("alpha_0")
        angle_list = alpha_0 + np.linspace(0, 2.0 * np.pi, n, endpoint=False)
        xs = self.get_param("x") + self.get_param("a") * np.cos(angle_list)
        ys = self.get_param("y") + self.get_param("b") * np.sin(angle_list)
        coords = list(zip(xs, ys))
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

        xs = [center_x - half_width, center_x - half_width,
              center_x + half_width, center_x + half_width]
        ys = [center_y - half_height, center_y + half_height,
              center_y + half_height, center_y - half_height]

        coords = list(zip(xs, ys))

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

        for sweep_index in sweep_over:
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
    DEFAULT_PARAMS = {
        "rounding_precision": None
    }

    def _polygonize(self):
        rp = self.get_param("rounding_precision")
        children_polys = []

        for child in self.children:
            if rp is not None:
                child = child.apply(CoordinatesRounded, rounding_precision=rp)
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
        left = self.children[0]  # we don't want to flatten the left side to preserve its interior boundaries
        right = self.children[1].apply(Flattened)

        polys_left = left.polygonize()
        poly_right = right.polygonize()[0]

        result_polys = []

        for poly_left in polys_left:
            func = getattr(poly_left, self.get_param("operation"))
            result_polys.append(func(poly_right))

        return result_polys


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


class BufferedShape(Shape):
    DEFAULT_PARAMS = {
        "d": 0.0,
        "resolution": 16,
        "cap_style": shapely.geometry.CAP_STYLE.round,
        "join_style": shapely.geometry.JOIN_STYLE.round,
        "mitre_limit": 5.0
    }

    def _polygonize(self):
        own_poly = Flattened(self.children).polygonize()[0]

        d, resolution, cap_style, join_style, mitre_limit = self.get_params(
            "d", "resolution", "cap_style", "join_style", "mitre_limit")

        buffered_poly = own_poly.buffer(
            distance=d,
            resolution=resolution,
            cap_style=cap_style,
            join_style=join_style,
            mitre_limit=mitre_limit
        )

        return [buffered_poly]


class PartitionedByLine(Shape):
    DEFAULT_PARAMS = {
        "origin": (0.0, 0.0),
        "direction": (1.0, 0.0)
    }

    def _polygonize(self):
        origin = np.array(self.get_param("origin"))
        direction = np.array(self.get_param("direction"))
        if np.linalg.norm(direction) == 0.0:
            raise ValueError("direction vector must have length > 0")
        direction /= np.linalg.norm(direction)

        bounding_box = np.array(Flattened(self.children).get_bounding_box())
        bbox_rect = shapely.geometry.box(*bounding_box)
        if not bbox_rect.intersects(shapely.geometry.Point(origin)):
            raise ValueError("origin must be inside the bounding box "
                "of the shape to be partitioned.")

        # draw a line that is guaranteed to be long enough
        # to fully cut through the bounding box
        # by making it four times as long as the box diagonal
        box_diagonal = np.hypot(np.diff(bounding_box[::2]),
                                np.diff(bounding_box[1::2]))[0]

        pos_partition_end = origin + box_diagonal*direction*2
        neg_partition_end = origin - box_diagonal*direction*2

        part_line = shapely.geometry.LineString(
            [neg_partition_end, pos_partition_end])

        # construct a list of edges of the bounding box
        bbox_coords = list(bbox_rect.exterior.coords)
        bbox_edges = np.array(list(zip(bbox_coords, bbox_coords[1:])))

        # check where the partioning line intersects the bounding box edges
        edge_intersects = []
        for edge_coords in bbox_edges:
            edge_line = shapely.geometry.LineString(edge_coords)
            edge_intersects.append(edge_line.intersection(part_line))

        # extract which edges are intersected
        intersecting_edges = np.array(
            [not isect.is_empty for isect in edge_intersects]
        )

        # find out if we have any intersection points that coincide
        # with two edges in the same point i.e. intersections exactly at
        # the corner of the bounding box
        edge_intersect_coords = np.array(
            [ei.coords[0] if has_ei else [np.nan, np.nan]
             for has_ei, ei in zip(intersecting_edges, edge_intersects)]
        )

        _edge_intersects_coords = np.roll(edge_intersect_coords, -1, axis=0)
        ei_dists = np.linalg.norm(
            edge_intersect_coords - _edge_intersects_coords,
            axis=1,
        )

        ei_coincides = np.isclose(ei_dists, 0)

        if intersecting_edges.sum() - ei_coincides.sum() != 2:
            raise ValueError(
                "partitioning line must intersect the exterior of the bounding box")

        first_intersecting_edge = np.argmax(intersecting_edges)
        # check if this intersection is shared with the following edge,
        # if so take that edge as the starting edge
        if ei_coincides[first_intersecting_edge]:
            first_intersecting_edge += 1

        first_intersection_point = np.array(
            [edge_intersect_coords[first_intersecting_edge]]
        )

        last_intersecting_edge = (
            len(intersecting_edges) - 1 - np.argmax(intersecting_edges[::-1])
        )
        # check if this intersection is shared with the previous edge, if so
        # take that edge as the final edge
        if ei_coincides[last_intersecting_edge-1]:
            last_intersecting_edge -= 1
        last_intersection_point = np.array(
            [edge_intersect_coords[last_intersecting_edge]])

        # construct a list of intermediate vertices between the edge
        # intersections, along the bounding box
        intermediate_vertices = bbox_edges[first_intersecting_edge:last_intersecting_edge, 1, :]
        partitioning_poly_vertices = np.vstack(
            (first_intersection_point,
             intermediate_vertices,
             last_intersection_point)
        )

        # construct a polygon that can be used for partitioning
        # i.e. partition the shape in stuff inside this polygon and
        # stuff outside this polygon
        partitioning_poly = shapely.geometry.Polygon(
            partitioning_poly_vertices)

        partitioning_funcs = [
            lambda poly_in: poly_in.intersection(partitioning_poly),
            lambda poly_in: poly_in.difference(partitioning_poly)
        ]

        partitioned_polys = []

        c_polys = []
        for c in self.children:
            c_polys += c.polygonize()

        for c_poly in c_polys:
            for pf in partitioning_funcs:
                partitioned_poly = pf(c_poly)
                if not partitioned_poly.is_empty:
                    partitioned_polys.append(partitioned_poly)

        return partitioned_polys


class CoordinatesRounded(Shape):
    DEFAULT_PARAMS = {
        "rounding_precision": 1
    }

    def _polygonize(self):
        children_polys = []

        for child in self.children:
            cps = child.polygonize()
            cps = [
                shapely.wkt.loads(
                    shapely.wkt.dumps(p, rounding_precision=self.get_param("rounding_precision"))
                ) 
                for p in cps
            ]
            children_polys += cps

        return children_polys
