from plugins.Equalizer_4__3.tools import (
    create_equalization_settings_dialog,
    equalX,
    equalY,
)
from base_plugin import BasePlugin


class Plugin(BasePlugin):
    def __init__(self, main_window, plugin_name):
        super().__init__(main_window, plugin_name)

    def execute(self):
        result = self.request_gui(create_equalization_settings_dialog(self.main_window))
        print(f"{result = }")
