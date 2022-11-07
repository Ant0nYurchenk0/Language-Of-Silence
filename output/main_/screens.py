from model import *
from object_detection.utils import visualization_utils as viz_utils
from kivymd.uix.label import MDLabel
from kivymd.uix.screen import MDScreen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.toolbar import MDToolbar, MDBottomAppBar
from kivymd.uix.list import OneLineAvatarListItem, ImageLeftWidget, MDList
from kivy.clock import Clock
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.lang.builder import Builder
from kivy.graphics.texture import Texture
from kivy.uix.scrollview import ScrollView

class MenuScreen(MDScreen):
    def go_to_dict(self):
        self.parent.transition.direction = "left"
        self.parent.current = "dictionary"
    def go_to_detect(self):
        self.parent.transition.direction = "left"
        self.parent.current = "detect"
    def exit(self):
        exit()

class DictionaryScreen(MDScreen):
    def init_dict(self):
        self.dict = {"Hello" : "logo.png" ,
                    "Thank You" : "logo.png",
                    "Yes" : "logo.png",
                    "No" : "logo.png",
                    "I Love You" : "logo.png",
        }

    def go_to_menu(self):
        self.parent.transition.direction = "right"
        self.parent.current = "menu"
    
    def build_list(self):
        if len(self.children) == 0:
            return
        self.init_dict()
        sv = ScrollView()
        list = MDList()
        sv.pos_hint = {"center_x":0.5, "center_y":0.39}
        sv.size_hint = (0.7, 1)
        sv.do_scroll = False, False
        sv.scroll_type = ["bars", "content"]
        for item in self.dict:
            sign = OneLineAvatarListItem()
            sign.text = item
            image = ImageLeftWidget(source = self.dict[item])
            sign.add_widget(image)
            list.add_widget(sign)
        sv.add_widget(list)
        self.add_widget(sv)

class DetectScreen(MDScreen):

    def go_to_menu(self):
        self.parent.transition.direction = "right"
        self.parent.current = "menu"
    
    def build_toolbar(self, layout):
        toolbar = MDToolbar( title = "Detect")
        toolbar.icon = "translate"
        toolbar.type = "bottom"
        toolbar.right_action_items = [["keyboard-backspace", lambda x: self.go_to_menu()]]
        bottombar = MDBottomAppBar()
        bottombar.add_widget(toolbar)
        layout.add_widget(bottombar)


    def detect(self):
        if len(self.children) != 0:
            return
        self.img1=Image()
        layout = MDBoxLayout(orientation = "vertical")
        layout.size_x = Window.width
        layout.add_widget(self.img1)
        #opencv2 stuffs

        Clock.schedule_interval(self.update, 1.0/33.0)
        self.build_toolbar(layout)
        self.add_widget(layout)
    
    def update(self, dt):
        global detections, frame, ret, image_np, input_tensor, capture
        ret, frame = capture.read()
        image_np = np.array(frame)
    
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop("num_detections"))
        detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
        detections["num_detections"] = num_detections

        # detection_classes should be ints.
        detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                        image_np_with_detections,
                        detections["detection_boxes"],
                        detections["detection_classes"]+label_id_offset,
                        detections["detection_scores"],
                        category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=5,
                        min_score_thresh=.5,
                        agnostic_mode=False)
        buf1 = cv2.flip(image_np_with_detections, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(image_np_with_detections.shape[1], image_np_with_detections.shape[0]), colorfmt='bgr') 
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.img1.texture = texture1