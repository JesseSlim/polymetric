from collections.abc import Mapping, MutableMapping
import inspect


class DerivedParameter:
    def __init__(self, deriving_function, description=""):
        self.deriving_function = deriving_function
        self.description = description

    def get(self, paramtree):
        return self.deriving_function(
            lambda key, paramtree=paramtree: paramtree[key]
        )

    def __repr__(self):
        df_string = "unknown"
        try:
            df_string = inspect.getsource(self.deriving_function)
            df_string = repr(df_string.strip())[1:-1]  # get rid of outer quote marks
        except:
            pass

        return("DerivedParameter(description='{}', deriving_function=<{}>)".format(self.description, df_string))


class ParameterTree(MutableMapping):
    def __init__(self, map_in=None, parent=None):
        if map_in is None:
            map_in = dict()

        # if not isinstance(map_in, MutableMapping):
        #     return ValueError("Initial mapping must be mutable")

        if not check_key_types(map_in):
            return TypeError("Initial dictionary contains non-string keys")

        # wrap all sub-dicts in ParameterTree instances
        self._data = self.wrap_submaps(map_in)
        self._parent = parent


    def wrap_submaps(self, map_in):
        wrapped_map = dict()
        for k, v in map_in.items():
            if isinstance(v, Mapping):
                wrapped_map[k] = ParameterTree(v, parent=self)
            else:
                wrapped_map[k] = v
        return wrapped_map

    def get_root(self):
        if self._parent is None:
            return self
        else:
            return self._parent.get_root()

    def __getitem__(self, key, is_subquery=False, derive_parameter=True):
        try:
            if not isinstance(key, str):
                raise TypeError("Key has to be string")
            if key == '':
                raise TypeError("The empty string is not a valid key")

            path_parts = key.split("/")
            first_part = path_parts.pop(0)

            if first_part == '':
                # the first character was a '/', find the root and traverse down from there
                subquery = "/".join(path_parts)
                return self.get_root().__getitem__(subquery, is_subquery=True, derive_parameter=derive_parameter)

            subitem = None
            if first_part == ".":
                subitem = self
            elif first_part == "..":
                subitem = self._parent
            else:
                subitem = self._data[first_part]

            if len(path_parts) == 0:
                if derive_parameter and isinstance(subitem, DerivedParameter):
                    return subitem.get(self)
                else:
                    return subitem
            else:
                subquery = "/".join(path_parts)
                return subitem.__getitem__(subquery, is_subquery=True, derive_parameter=derive_parameter)
        except KeyError as e:
            if is_subquery:
                raise e

        raise KeyError(key)

    def get(self, key, default=None, derive_parameter=True):
        'D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.'
        try:
            return self.__getitem__(key, derive_parameter=derive_parameter)
        except KeyError:
            return default

    def __setitem__(self, key, value, is_subquery=False):
        try:
            if not isinstance(key, str):
                raise TypeError("Key has to be string")
            if key == '':
                raise TypeError("The empty string is not a valid key")

            path_parts = key.split("/")
            first_part = path_parts.pop(0)

            if first_part == '':
                # the first character was a '/', find the root and traverse down from there
                subquery = "/".join(path_parts)
                return self.get_root().__setitem__(subquery, value, is_subquery=True)

            if len(path_parts) > 0:
                subitem = None
                if first_part == ".":
                    subitem = self
                elif first_part == "..":
                    subitem = self._parent
                else:
                    subitem = self._data[first_part]

                subquery = "/".join(path_parts)
                return subitem.__setitem__(subquery, value, is_subquery=True)
            else:
                if first_part == '.' or first_part == '..':
                    raise TypeError(". and .. are invalid leaf keys")

                if isinstance(value, Mapping):
                    value = ParameterTree(value, parent=self)

                self._data[first_part] = value
                return
        except KeyError as e:
            if is_subquery:
                raise e

        raise KeyError(key)

    def __delitem__(self, key, is_subquery=False):
        try:
            if not isinstance(key, str):
                raise TypeError("Key has to be string")
            if key == '':
                raise TypeError("The empty string is not a valid key")

            path_parts = key.split("/")
            first_part = path_parts.pop(0)

            if first_part == '':
                # the first character was a '/', find the root and traverse down from there
                subquery = "/".join(path_parts)
                return self.get_root().__delitem__(subquery, is_subquery=True)

            if len(path_parts) > 0:
                subitem = None
                if first_part == ".":
                    subitem = self
                elif first_part == "..":
                    subitem = self._parent
                else:
                    subitem = self._data[first_part]

                subquery = "/".join(path_parts)
                return subitem.__delitem__(subquery, is_subquery=True)
            else:
                if first_part == "." or first_part == "..":
                    raise TypeError(". and .. are invalid leaf keys")

                del self._data[first_part]
                return
        except KeyError as e:
            if is_subquery:
                raise e

        raise KeyError(key)

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def todict(self, derive_parameter=True):
        result_dict = dict()

        for k in self:
            v = self.get(k, derive_parameter=derive_parameter)
            if isinstance(v, ParameterTree):
                result_dict[k] = v.todict(derive_parameter=derive_parameter)
            else:
                result_dict[k] = v

        return result_dict

    def __repr__(self):
        return "ParameterTree({})".format(str(self.todict(derive_parameter=False)))


def check_key_types(map_in):
    for k, v in map_in.items():
        if not isinstance(k, str):
            return False

        if isinstance(v, Mapping):
            if not check_key_types(v):
                return False

    return True


