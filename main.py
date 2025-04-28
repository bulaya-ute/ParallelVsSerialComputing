from kivy.metrics import dp
from kivy.properties import ListProperty, BooleanProperty, VariableListProperty, ColorProperty, OptionProperty, Clock, \
    NumericProperty
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivymd.theming import ThemableBehavior
from kivymd.tools.hotreload.app import MDApp
from kivymd.uix import MDAdaptiveWidget
from kivymd.uix.behaviors import DeclarativeBehavior, BackgroundColorBehavior, CommonElevationBehavior, \
    RectangularRippleBehavior
from kivymd.uix.behaviors.state_layer_behavior import StateLayerBehavior
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.card import MDCard
from kivymd.uix.progressindicator import MDLinearProgressIndicator
from kivymd.uix.screen import MDScreen


inputs = []  # Populate this list with the inputs you want to process


def processor(_input: float | int):
    """
    Implement this and delete this docstring when done
    It should return a float or int, after performing an operation on the input value
    """


class App(MDApp):
    DEBUG = True
    AUTORELOADER_PATHS = [
        (".", {"recursive": False}),
        ("./KV", {"recursive": True}),
    ]
    KV_DIRS = [
        "./KV"
    ]

    def build_app(self, first=False):
        return MainScreen()

    def start_serial_processing(self):
        """Implement this and delete this docstring when done"""

    def start_parallel_processing(self):
        """Implement this and delete this docstring when done"""


class MainScreen(MDScreen):
    pass


class SingleProgressBar(MDBoxLayout):
    indicator_color = ColorProperty([1, 0, 1, 1])
    value = NumericProperty(0.0)


class ProcessingProgressBar(MDBoxLayout):
    values = ListProperty([0])
    indicator_color = ColorProperty([1, 0, 1, 1])
    indicator_bg_color = ColorProperty([1, 0, 0, 1])

    def on_values(self, instance, values):
        values = self.values

        # Ensure the correct number of progress bars are present.
        while len(self.children) != len(values):
            if len(self.children) < len(values):
                self.add_widget(SingleProgressBar(
                    # md_bg_color=[0, 0, 1, 1],
                    # indicator_color=self.indicator_color,
                ))
            elif len(self.children) > len(values):
                self.remove_widget(self.children[-1])

        for i, progress_bar in enumerate(self.children):
            progress_bar.value = self.values[i]


class TimeIndicator(MDBoxLayout):
    time_elapsed = NumericProperty(0.0)
    processing_time = NumericProperty(0.0)


if __name__ == "__main__":
    App().run()