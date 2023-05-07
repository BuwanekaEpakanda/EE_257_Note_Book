from ipywidgets import interact, interactive
import ipywidgets as widgets
from typing import Dict, Any, List, Callable, Tuple
import matplotlib.patches as mpatches

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display, HTML


class Variable:
    def __init__(self, name, value):
        self.name = name
        self.value: Any = value

    def __call__(self) -> Any:
        return self.value

    def update_value(self, new_value: Any):
        self.value = new_value

    def update_on_change(self, change):
        self.update_value(change["new"])


class RealTimeFunction:
    def __init__(self, time_function: Callable[[np.ndarray], np.ndarray], vrange: Tuple[float, float], name: str, points=100, ):
        '''
        time_function   :   any function that operates on a 1 dimensional numpy array (time linspace) 
                            and returns a one dimensional array (x(t)) of real numbers. Models a function of
                            the form 
        vrange           :   Range of values to plot the graph 
        points          :   number of points to plot
        '''
        self.time_function = time_function
        self.time_vals = np.linspace(
            vrange[0], vrange[1], points)
        self.name = name

    def __call__(self):
        return self.time_function(self.time_vals)

    def draw(self, ax):
        ax.plot(self.time_vals, self.__call__(), label=self.name)


class ComplexTimePoint:
    def __init__(self, time_function: Callable[[], np.complex_], name: str, size=5):
        '''
        time_function   : should return a complex number as point 

        Plots a single point on the plot considering the plot as an argand diagram
        '''
        self.time_function = time_function
        self.name = name
        self.size = size

    def __call__(self):
        return self.time_function()

    def draw(self, ax):
        val = self()
        ax.plot(val.real, val.imag, marker="o", markersize=self.size,
                markerfacecolor="green", label=self.name)


def complex_to_tuple(z: np.complex_):
    return (z.real, z.imag)


class Circle:
    def __init__(self, radius_func: Callable[[], float], center_func: Callable[[], Tuple[float, float]]):
        self.radius_func = radius_func
        self.center_func = center_func

    def draw(self, ax):
        circle = mpatches.Circle(
            tuple(self.center_func()), self.radius_func(), color='r', fill=False, linestyle="--")
        ax.add_patch(circle)


class ClearComplexPoint:
    def __init__(self, time_function: Callable[[], np.complex_], name: str, size=5):
        '''
        time_function   : should return a complex number as point 

        Plots a single point on the plot considering the plot as an argand diagram
        '''
        self.time_function = time_function
        self.name = name
        self.size = size

    def __call__(self):
        return self.time_function()

    @staticmethod
    def draw_static(ax, start: np.complex_, end: np.complex_, size):
        ax.plot(end.real, end.imag, marker="o", markersize=size,
                markerfacecolor="green")
        circle = mpatches.Circle(
            (start.real, start.imag), np.abs(end-start), color='r', fill=False, linestyle="--")  # type: ignore
        ax.add_patch(circle)
        x_ends = [start.real, end.real]
        y_ends = [start.imag, end.imag]
        ax.plot(x_ends, y_ends, linestyle="--", color="blue")

    def draw(self, ax):
        val = self()
        self.draw_static(ax, 0+0j, val, size=self.size)  # type: ignore


class VerticalRealTimeFunction(RealTimeFunction):
    def draw(self, ax):
        ax.plot(self.__call__(), self.time_vals,  label=self.name)


class ComplexPointChain:
    def __init__(self, time_functions: List[Callable[[], np.complex_]], name: str, size=5):
        '''
        time_function   : should return a complex number as point 

        Plots a single point on the plot considering the plot as an argand diagram
        '''
        self.time_functions = time_functions
        self.name = name
        self.size = size

    def __call__(self):
        old_point = 0.0+0.0j
        for f in self.time_functions:
            delta_point = f()
            old_point += delta_point
        return old_point

    def draw(self, ax):
        old_point = 0.0+0.0j
        for f in self.time_functions:
            delta_point = f()
            new_point = old_point+delta_point
            ClearComplexPoint.draw_static(
                ax, old_point, new_point, self.size)  # type: ignore
            old_point = new_point


class Graphic:
    def __init__(self, figsize: Tuple[int, int] = (5, 5)):
        self.widgets = []
        self.variables: List[Variable] = []
        self.output = widgets.Output()
        self.drawables = []
        self.limits = None
        self.figsize = figsize

    def set_limits(self, xmin, xmax, ymin, ymax):
        self.limits = (xmin, xmax, ymin, ymax)

    def on_update(self, change):
        with self.output:

            self.axes = plt.figure(figsize=self.figsize).add_subplot(111)
            if(self.limits is not None):
                plt.xlim(self.limits[0], self.limits[1])
                plt.ylim(self.limits[2], self.limits[3])
            self.output.clear_output(wait=True)

            for drawable in self.drawables:
                drawable.draw(self.axes)
            # plt.legend(loc="upper left")
            plt.show()

    def add_drawable(self, drawable):
        self.drawables.append(drawable)

    def show(self):
        display(self.output, *self.widgets)
        self.on_update(None)

    def get_slider_variable(self,
                            default_value: float, min_val: float, max_val: float, step: float,
                            name: str,
                            lazy=False):
        slider = widgets.FloatSlider(
            value=default_value,
            min=min_val,
            max=max_val,
            step=step,
            description=f"{name}: "
        )
        variable = Variable(name, slider.get_interact_value(),)
        slider.observe(variable.update_on_change,
                       names='value')  # type: ignore
        if(not lazy):
            slider.observe(self.on_update, names='value')  # type: ignore
        self.widgets.append(slider)
        self.variables.append(variable)
        return variable

    def animated_slider(self,
                        default_value: float, min_val: float, max_val: float, step: float,
                        name: str,
                        ):

        play = widgets.Play(
            value=default_value,
            min=min_val/step,
            max=max_val/step,
            step=1,
            interval=10,
            description="Press play",
            disabled=False
        )
        slider = widgets.FloatSlider(
            value=default_value,
            min=min_val,
            max=max_val,
            step=step,
            description=f"{name}: "
        )

        variable = Variable(name, default_value)
        slider.observe(variable.update_on_change,
                       names='value')  # type: ignore

        def set_slider(x):
            slider.set_trait('value', x["new"]*step)
            # slider.value = x["new"]*step

        play.observe(set_slider, names='value')  # type: ignore
        slider.observe(self.on_update, names='value')  # type: ignore
        self.widgets.append(play)
        self.widgets.append(slider)
        self.variables.append(variable)
        return variable
