from screens import *
from kivymd.app import MDApp
from kivy.uix.screenmanager import ScreenManager

class LOSApp(MDApp):
    sm = ScreenManager()
    sm.add_widget(MenuScreen(name = "menu"))
    sm.add_widget(DictionaryScreen(name = "dictionary"))
    sm.add_widget(DetectScreen(name = "detect"))
    

    def build(self):
        screen = Builder.load_file("LOS.kv")
        return screen
