import numpy as np


valid_sub_type:dict = {
    'a': ['n', 'number',],
    'b': ['s', 'total',],
    'c': ['d', 'each', 'unit'],
}


class Point2D:

    def __init__(
        self,
        x:int or float=np.int64(0),
        y:int or float=np.int64(0),
        axis_rotation_angle:np.float64=np.float64(0),
        # coord_type:str=None,
    ) -> None:

        valid_dtype:tuple = (
            'int64', 'float64',
            'int', 'float',
        )
        if any(
            type(param).__name__ not in valid_dtype
            for param in [x, y]
        ):
            raise TypeError(
                'Please check data type to set a point  ' + \
                f'x: {type(x)}, y: {type(y)}'
            )

        final_x = x
        final_y = y
        if axis_rotation_angle!=0:
            cosine = np.cos(axis_rotation_angle, dtype=np.float64)
            sine = np.sin(axis_rotation_angle, dtype=np.float64)
            final_x = np.float64(x*cosine-y*sine)
            final_y = np.float64(x*sine+y*cosine)

        self.x = final_x
        self.y = final_y

            
class Voxel:
    
    def __init__(
        self,
        dtype:str,
        x:int or float,
        y:int or float,
        # z:int or float=None,
        **kw
    ) -> None:

        self.arr:list = None

        if all(
            dtype not in val
            for val in valid_sub_type.values()
        ):
            raise TypeError(
                f"{dtype} is invalid, choose one of here;" + \
                    f" {valid_sub_type}"
            )

        else:
            if dtype in valid_sub_type['a']:
                self.x = np.int64(x) 
                self.y = np.int64(y) 
                # if z is None:
                #     self.z = np.int64(1)
                # else:
                #     self.z = np.int64(z) 

            if dtype in valid_sub_type['b']:
                self.x = np.float64(x) 
                self.y = np.float64(y) 
                # if z is None:
                #     self.z = np.int64(1)
                # else:
                #     self.z = np.int64(z) 

            if dtype in valid_sub_type['c']:
                self.x = np.float64(x) 
                self.y = np.float64(y) 
                # if z is None:
                #     self.z = np.int64(1)
                # else:
                #     self.z = np.int64(z) 

        # self.arr = np.array([self.z, self.y, self.x])
        self.arr = np.array([self.y, self.x])


class Detector:
    
    def __init__(
        self,
        dtype:str,
        u:int or float,
        # v:int or float=None,
        # **kw
    ) -> None:

        self.arr:list = None

        if all(
            dtype not in val
            for val in valid_sub_type.values()
        ):
            raise TypeError(
                f"{dtype} is invalid, choose one of here;" + \
                    f" {valid_sub_type}"
            )

        else:
            if dtype in valid_sub_type['a']:
                self.u = np.int64(u) 
                # if v is None:
                #     self.v = np.int64(1)
                # else:
                #     self.v = np.int64(v) 

            if dtype in valid_sub_type['b']:
                self.u = np.float64(u) 
                # if v is None:
                #     self.v = np.float64(1)
                # else:
                #     self.v = np.float64(v) 

            if dtype in valid_sub_type['c']:
                self.u = np.float64(u) 
                # if v is None:
                #     self.v = np.float64(1)
                # else:
                #     self.v = np.float64(v) 

        # self.arr = np.array([self.v, self.u])
        self.arr = np.array([self.u])


class Vector2D:
    
    def __init__(
        self,
        start_point:Point2D,
        end_point:Point2D,
    ) -> None:

        _dx:np.float64 = np.float64(end_point.x)-np.float64(start_point.x)
        _dy:np.float64 = np.float64(end_point.y)-np.float64(start_point.y)

        self.size:np.float64 = np.sqrt(
            _dx**2+_dy**2,
            dtype=np.float64
        )

        self.vector:tuple = tuple([_dx, _dy])

        self.unit:tuple = (
            np.float64(self.vector[0]/self.size),
            np.float64(self.vector[1]/self.size)
        )
